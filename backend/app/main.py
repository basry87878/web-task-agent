from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from app.agent import get_cached_research, normalize_prompt_cache_key, run_research
from app.coach import (
    CoachInputError,
    CoachUpstreamError,
    CoachUpstreamTimeoutError,
    ask_prompt_coach,
)
from app.config import (
    API_IP_RATE_LIMIT_PER_MINUTE,
    ENABLE_API_DOCS,
    MAX_REQUEST_BODY_BYTES,
    MAX_RESEARCH_PROMPT_CHARS,
    MAX_RESEARCH_RUNS_PER_SESSION,
    OPENROUTER_TIMEOUT_SECONDS,
    PRESETS,
    PRESETS_BY_ID,
    RATE_LIMIT_TTL_SECONDS,
    RUN_COOLDOWN_SECONDS,
    get_cors_origins,
    get_ga_api_secret,
    get_ga_measurement_id,
)
from app.deep_research import (
    get_export_file_bytes,
    get_job,
    request_stop_job,
    start_deep_research_job,
)
from app.exporter import (
    export_rows_to_csv_bytes,
    export_rows_to_excel_bytes,
    make_safe_filename,
    validate_export_rows,
)
from app.guardrails import (
    SessionRateLimiter,
    SlidingWindowIPRateLimiter,
    classify_research_intent,
    redact_secrets,
    validate_user_input,
)
from app.providers.deep_router import get_provider_status
from app.schemas import (
    CoachRequest,
    CoachResponse,
    DeepResearchStartRequest,
    DeepResearchStartResponse,
    DeepResearchStatusResponse,
    ExportRequest,
    PresetOut,
    PresetsResponse,
    ResearchRequest,
    ResearchResponse,
)

logger = logging.getLogger("jobtruth.api")

app = FastAPI(
    title="JobTruth AI Market Research API",
    version="2.0.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    expose_headers=["Content-Disposition"],
)

research_limiter = SessionRateLimiter(
    max_count=MAX_RESEARCH_RUNS_PER_SESSION,
    ttl_seconds=RATE_LIMIT_TTL_SECONDS,
)
api_ip_limiter = SlidingWindowIPRateLimiter(
    max_requests=API_IP_RATE_LIMIT_PER_MINUTE,
    window_seconds=60,
)


def _extract_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


COACH_ENDPOINT_TIMEOUT_SECONDS = max(OPENROUTER_TIMEOUT_SECONDS + 15.0, 35.0)


def _apply_security_headers(response):
    """Attach mandatory HTTP security headers to every API response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains",
    )
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


@app.middleware("http")
async def security_hardening_middleware(request: Request, call_next):
    # 1. Enforce strict request body size limit (prevents memory/DoS exhaustion)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return _apply_security_headers(
                    JSONResponse(
                        status_code=413,
                        content={"error": "payload_too_large", "detail": "Request payload too large."},
                    )
                )
        except ValueError:
            return _apply_security_headers(
                JSONResponse(
                    status_code=400,
                    content={"error": "invalid_header", "detail": "Invalid Content-Length header."},
                )
            )

    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > MAX_REQUEST_BODY_BYTES:
            return _apply_security_headers(
                JSONResponse(
                    status_code=413,
                    content={"error": "payload_too_large", "detail": "Request payload too large."},
                )
            )

    # 2. Enforce IP-based rate limiting across all /api/ endpoints (independent of session_id)
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        client_ip = _extract_client_ip(request)
        allowed, retry_after = api_ip_limiter.allow(client_ip)
        if not allowed:
            return _apply_security_headers(
                JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limited",
                        "detail": "Too many requests from this IP. Please slow down and retry shortly.",
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            )

    response = await call_next(request)

    # 3. Attach mandatory defensive security headers
    return _apply_security_headers(response)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a clean structured JSON 400 error on malformed JSON or schema validation failures."""
    return _apply_security_headers(
        JSONResponse(
            status_code=400,
            content={
                "error": "invalid_input",
                "detail": "Malformed or invalid request payload. Please check your input and try again.",
                "retryable": False,
            },
        )
    )


@app.exception_handler(HTTPException)
async def sanitized_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure HTTP 4xx/5xx error responses never leak API keys, credentials, or raw tracebacks."""
    raw_detail = exc.detail if isinstance(exc.detail, str) else "Request could not be completed."
    safe_detail = redact_secrets(raw_detail)
    if "Traceback" in safe_detail or 'File "' in safe_detail:
        safe_detail = "The service could not complete your request. Please try again shortly."
    return _apply_security_headers(
        JSONResponse(
            status_code=exc.status_code,
            content={"detail": safe_detail},
            headers=exc.headers,
        )
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unexpected exceptions so stack traces and secrets never leak to clients."""
    logger.error("Unhandled API error on %s: %s", request.url.path, redact_secrets(repr(exc)))
    return _apply_security_headers(
        JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )
    )


def _client_key(request: Request, session_id: str | None) -> str:
    ip = _extract_client_ip(request)
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

    # Resolve preset or run pre-search intent & arithmetic/injection gate before any API call
    is_preset = bool(body.preset_id and body.preset_id in PRESETS_BY_ID)
    if is_preset and body.preset_id:
        prompt_to_run = PRESETS_BY_ID[body.preset_id]["prompt"]
        preset_id = body.preset_id
    else:
        preset_id = None
        decision = classify_research_intent(
            body.prompt,
            max_chars=MAX_RESEARCH_PROMPT_CHARS,
            min_chars=3,
        )
        if decision.action in ("block_injection", "invalid"):
            raise HTTPException(status_code=400, detail=decision.message)
        if decision.action == "short_circuit":
            # Return clean 0-row out-of-scope response immediately without burning Tavily/LLM/quota
            title = "Direct Calculation" if decision.calculation_result else "Out of Scope for B2B Research"
            return ResearchResponse(
                title=title,
                executive_summary=decision.message,
                columns=["Name", "Category", "Key Details", "Pricing / Metrics", "Notes", "Source URL"],
                rows=[],
                flagged_rows=[],
                model_used="Intent Guardrail",
                verified_url_count=0,
                unverified_dropped=0,
                duration_sec=0.0,
                cached=False,
                filename=make_safe_filename(title, ext="xlsx"),
                runs_remaining=research_limiter.remaining(key),
            )
        prompt_to_run = decision.cleaned_prompt

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
        logger.warning("Research pipeline failure: %s", redact_secrets(repr(exc)))
        raise HTTPException(
            status_code=502,
            detail="The research providers are temporarily busy or unavailable. Please try again in a moment.",
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
    """Generate an in-memory .xlsx or .csv attachment strictly from Pydantic-validated rows (zero disk writes)."""
    validated_rows = validate_export_rows(body.rows)
    if not validated_rows:
        raise HTTPException(
            status_code=400,
            detail="No valid, grounded rows are available to export.",
        )

    if body.format == "csv":
        data = export_rows_to_csv_bytes(validated_rows)
        filename = make_safe_filename(body.title, ext="csv")
        media_type = "text/csv; charset=utf-8"
        track_ga_event("csv_download", {"rows_count": len(validated_rows)})
    else:
        data = export_rows_to_excel_bytes(body.title, body.executive_summary, validated_rows)
        filename = make_safe_filename(body.title, ext="xlsx")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        track_ga_event("excel_download", {"rows_count": len(validated_rows)})

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/coach", response_model=CoachResponse)
async def api_coach(body: CoachRequest, request: Request) -> CoachResponse | JSONResponse:
    msg_dicts = [{"role": m.role, "content": m.content} for m in body.messages]
    t0 = time.perf_counter()
    try:
        reply_text, suggested_prompt, model_used = await asyncio.wait_for(
            asyncio.to_thread(ask_prompt_coach, msg_dicts),
            timeout=COACH_ENDPOINT_TIMEOUT_SECONDS,
        )
    except (CoachInputError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_input",
                "detail": redact_secrets(str(exc)),
                "retryable": False,
            },
        )
    except (CoachUpstreamTimeoutError, asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("Prompt Coach upstream timeout: %s", redact_secrets(repr(exc)))
        return JSONResponse(
            status_code=504,
            content={
                "error": "upstream_timeout",
                "detail": "Prompt Coach timed out while contacting the upstream model. Please try again in a moment.",
                "retryable": True,
            },
        )
    except (CoachUpstreamError, Exception) as exc:
        logger.warning("Prompt Coach upstream failure: %s", redact_secrets(repr(exc)))
        return JSONResponse(
            status_code=503,
            content={
                "error": "upstream_unavailable",
                "detail": "Prompt Coach is temporarily unavailable. Please try again shortly.",
                "retryable": True,
            },
        )

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


# ---------------------------------------------------------------------------
# Deep Research Async Job Endpoints (Strictly Isolated from Fast Mode / Tavily)
# ---------------------------------------------------------------------------


@app.get("/api/deep-research/providers")
def api_deep_research_providers() -> dict[str, object]:
    """Safe boolean readiness check showing which Deep Research providers are configured (never exposes keys)."""
    return get_provider_status()


@app.post("/api/deep-research", response_model=DeepResearchStartResponse)
async def api_start_deep_research(
    body: DeepResearchStartRequest,
    request: Request,
) -> DeepResearchStartResponse:
    try:
        job = start_deep_research_job(body.prompt, target_rows=body.target_rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(str(exc))) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=redact_secrets(str(exc))) from exc

    track_ga_event(
        "deep_research_started",
        {"target_rows": body.target_rows or 0},
        client_id=body.session_id,
    )
    return DeepResearchStartResponse(
        job_id=job.job_id,
        status=job.status if job.status == "completed" else "running",
        active_providers=job.active_providers,
        target_rows=job.target_rows,
    )


@app.get("/api/deep-research/{job_id}", response_model=DeepResearchStatusResponse)
def api_deep_research_status(job_id: str) -> DeepResearchStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Deep Research job not found.")
    return DeepResearchStatusResponse(**job.to_status_dict())


@app.post("/api/deep-research/{job_id}/stop", response_model=DeepResearchStatusResponse)
def api_deep_research_stop(job_id: str) -> DeepResearchStatusResponse:
    job = request_stop_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Deep Research job not found.")
    track_ga_event(
        "deep_research_stopped",
        {"rows_collected": len(job.rows), "wave": job.wave},
    )
    return DeepResearchStatusResponse(**job.to_status_dict())


@app.get("/api/deep-research/{job_id}/download")
def api_deep_research_download(
    job_id: str,
    format: str = Query(default="xlsx", pattern="^(xlsx|csv)$"),
) -> StreamingResponse:
    export_pkg = get_export_file_bytes(job_id, fmt=format)
    if export_pkg is None:
        raise HTTPException(
            status_code=404,
            detail="No exportable rows are ready yet for this Deep Research job.",
        )
    data, filename, media_type = export_pkg
    track_ga_event("deep_research_downloaded", {"format": format})
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
