from __future__ import annotations

import asyncio
import io
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent import get_cached_research, normalize_prompt_cache_key, run_research
from app.coach import ask_prompt_coach
from app.config import (
    MAX_RESEARCH_PROMPT_CHARS,
    MAX_RESEARCH_RUNS_PER_SESSION,
    PRESETS,
    PRESETS_BY_ID,
    RATE_LIMIT_TTL_SECONDS,
    RUN_COOLDOWN_SECONDS,
    get_cors_origins,
    get_ga_api_secret,
    get_ga_measurement_id,
)
from app.exporter import (
    export_rows_to_csv_bytes,
    export_rows_to_excel_bytes,
    make_safe_filename,
)
from app.guardrails import SessionRateLimiter, redact_secrets, validate_user_input
from app.schemas import (
    CoachRequest,
    CoachResponse,
    ExportRequest,
    PresetOut,
    PresetsResponse,
    ResearchRequest,
    ResearchResponse,
)

app = FastAPI(
    title="JobTruth AI Market Research API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

research_limiter = SessionRateLimiter(
    max_count=MAX_RESEARCH_RUNS_PER_SESSION,
    ttl_seconds=RATE_LIMIT_TTL_SECONDS,
)


def _client_key(request: Request, session_id: str | None) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    sid = (session_id or "").strip()[:64]
    return f"{ip}:{sid}" if sid else ip


def track_ga_event(
    event_name: str,
    params: dict[str, str | int | float | bool] | None = None,
    client_id: str | None = None,
) -> None:
    """Fire-and-forget GA4 Measurement Protocol event in a background daemon thread."""
    measurement_id = get_ga_measurement_id()
    api_secret = get_ga_api_secret()
    if not measurement_id or not api_secret:
        return

    cid = (client_id or "").strip() or str(uuid.uuid4())
    safe_params: dict[str, str | int | float | bool] = {"engagement_time_msec": 100}
    if params:
        for k, v in params.items():
            if isinstance(v, (bool, int, float)):
                safe_params[k] = v
            elif isinstance(v, str):
                safe_params[k] = redact_secrets(v)[:100]

    payload = {
        "client_id": cid,
        "events": [{"name": event_name, "params": safe_params}],
    }

    def _send() -> None:
        try:
            import httpx

            httpx.post(
                "https://www.google-analytics.com/mp/collect",
                params={"measurement_id": measurement_id, "api_secret": api_secret},
                json=payload,
                timeout=3.0,
            )
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health check used for silent Render cold-start wake-up."""
    return {"status": "ok"}


@app.get("/api/presets", response_model=PresetsResponse)
def get_presets(
    request: Request,
    session_id: str | None = Query(default=None, max_length=64),
) -> PresetsResponse:
    key = _client_key(request, session_id)
    return PresetsResponse(
        presets=[PresetOut(**p) for p in PRESETS],
        runs_remaining=research_limiter.remaining(key),
        replies_remaining=None,
        ga_measurement_id=get_ga_measurement_id(),
    )


@app.post("/api/research", response_model=ResearchResponse)
async def api_research(body: ResearchRequest, request: Request) -> ResearchResponse:
    key = _client_key(request, body.session_id)

    # Resolve preset or validate custom user prompt before any API call
    is_preset = bool(body.preset_id and body.preset_id in PRESETS_BY_ID)
    if is_preset and body.preset_id:
        prompt_to_run = PRESETS_BY_ID[body.preset_id]["prompt"]
        preset_id = body.preset_id
    else:
        preset_id = None
        is_valid, cleaned_or_err = validate_user_input(
            body.prompt,
            max_chars=MAX_RESEARCH_PROMPT_CHARS,
            min_chars=10,
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=cleaned_or_err)
        prompt_to_run = cleaned_or_err

    # Check 1-hour cache first — cache hits return immediately without burning quota or cooldown
    cache_key = normalize_prompt_cache_key(prompt_to_run, preset_id=preset_id)
    cached_outcome = get_cached_research(cache_key)
    if cached_outcome is not None:
        runs_rem = research_limiter.remaining(key)
        track_ga_event(
            "preset_run" if is_preset else "custom_run",
            {
                "preset_id": preset_id or "custom",
                "model_used": cached_outcome.model_used,
                "rows_count": len(cached_outcome.rows),
                "verified_urls": cached_outcome.verified_url_count,
                "duration_sec": cached_outcome.duration_sec,
                "cached": True,
            },
            client_id=body.session_id,
        )
        return ResearchResponse(
            title=cached_outcome.title,
            executive_summary=cached_outcome.executive_summary,
            columns=cached_outcome.columns,
            rows=cached_outcome.rows,
            flagged_rows=cached_outcome.flagged_rows,
            model_used=cached_outcome.model_used,
            verified_url_count=cached_outcome.verified_url_count,
            unverified_dropped=cached_outcome.unverified_dropped,
            duration_sec=cached_outcome.duration_sec,
            cached=True,
            filename=cached_outcome.filename,
            runs_remaining=runs_rem,
        )

    # Enforce server-side rate limit (3 uncached custom runs per session/IP) + anti-spam cooldown
    wait_secs = research_limiter.check_cooldown(key, RUN_COOLDOWN_SECONDS)
    if wait_secs > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {wait_secs}s before starting another research run.",
        )

    if not is_preset:
        allowed, rem = research_limiter.can_consume(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Session limit of {MAX_RESEARCH_RUNS_PER_SESSION} custom research runs reached. "
                    "You can still explore the cached business presets or export your results."
                ),
            )

    try:
        outcome = await asyncio.to_thread(
            run_research,
            prompt_to_run,
            preset_id=preset_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(str(exc))) from exc
    except Exception as exc:
        research_limiter.touch(key)
        raise HTTPException(
            status_code=502,
            detail=f"Could not complete research: {redact_secrets(str(exc))}",
        ) from exc

    if is_preset:
        research_limiter.touch(key)
        runs_rem = research_limiter.remaining(key)
    else:
        runs_rem = research_limiter.consume(key, touch_cooldown=True)

    track_ga_event(
        "preset_run" if is_preset else "custom_run",
        {
            "preset_id": preset_id or "custom",
            "model_used": outcome.model_used,
            "rows_count": len(outcome.rows),
            "verified_urls": outcome.verified_url_count,
            "duration_sec": outcome.duration_sec,
            "cached": outcome.cached,
        },
        client_id=body.session_id,
    )

    return ResearchResponse(
        title=outcome.title,
        executive_summary=outcome.executive_summary,
        columns=outcome.columns,
        rows=outcome.rows,
        flagged_rows=outcome.flagged_rows,
        model_used=outcome.model_used,
        verified_url_count=outcome.verified_url_count,
        unverified_dropped=outcome.unverified_dropped,
        duration_sec=outcome.duration_sec,
        cached=outcome.cached,
        filename=outcome.filename,
        runs_remaining=runs_rem,
    )


@app.post("/api/export")
def api_export(body: ExportRequest) -> StreamingResponse:
    """Generate an in-memory .xlsx or .csv attachment (zero disk writes, formula-safe)."""
    if body.format == "csv":
        data = export_rows_to_csv_bytes(body.rows)
        filename = make_safe_filename(body.title, ext="csv")
        media_type = "text/csv; charset=utf-8"
        track_ga_event("csv_download", {"rows_count": len(body.rows)})
    else:
        data = export_rows_to_excel_bytes(body.title, body.executive_summary, body.rows)
        filename = make_safe_filename(body.title, ext="xlsx")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        track_ga_event("excel_download", {"rows_count": len(body.rows)})

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/coach", response_model=CoachResponse)
async def api_coach(body: CoachRequest, request: Request) -> CoachResponse:
    msg_dicts = [{"role": m.role, "content": m.content} for m in body.messages]
    t0 = time.perf_counter()
    try:
        reply_text, suggested_prompt, model_used = await asyncio.to_thread(
            ask_prompt_coach, msg_dicts
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(str(exc))) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=redact_secrets(str(exc)),
        ) from exc

    elapsed = round(time.perf_counter() - t0, 2)
    track_ga_event(
        "prompt_coach_message",
        {
            "has_suggested_prompt": bool(suggested_prompt),
            "duration_sec": elapsed,
        },
        client_id=body.session_id,
    )

    return CoachResponse(
        reply=reply_text,
        suggested_prompt=suggested_prompt,
        model_used=model_used,
        replies_remaining=None,
    )
