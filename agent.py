from __future__ import annotations

from dataclasses import dataclass
import json
import random
import time
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from config import (
    AGENT_RECURSION_LIMIT,
    GEMINI_RETRY_ATTEMPTS,
    GROQ_REQUEST_TIMEOUT_SECONDS,
    GROQ_RETRY_ATTEMPTS,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
    MAX_EXTRACT_CALLS,
    MAX_EXTRACT_CHARS_PER_URL,
    MAX_EXTRACT_URLS,
    MAX_SEARCH_CALLS,
    MODEL_CHAIN,
    MODEL_RETRY_MAX_DELAY_SECONDS,
    MODEL_RETRY_MIN_DELAY_SECONDS,
    TAVILY_TIMEOUT_SECONDS,
    get_gemini_api_key,
    get_groq_api_key,
    get_tavily_api_key,
)
from schemas import (
    ResearchReport,
    export_report_to_excel_bytes,
    make_safe_filename,
    normalize_url,
    report_to_dataframe,
    verify_report_sources,
)

RESEARCH_SYSTEM_PROMPT = f"""You are a fast, rigorous B2B market research analyst. The current year is 2026.
Your goal is to gather accurate, up-to-date, source-verified facts from the live web using Tavily tools while keeping latency low.

RULES:
1. Always run 2–3 distinct `tavily_search` queries (hard cap: {MAX_SEARCH_CALLS}) covering complementary angles (e.g., latest 2026 financials/pricing, specific product tiers/segments/regions, and key competitors or official pages) to gather ~10–15 verified URLs before stopping. You may issue 2–3 focused searches in parallel to stay fast.
2. Always include "2026" (e.g., "FY 2026", "2026 pricing", "2026") in queries for financials, revenue, market figures, or pricing so you retrieve the newest data rather than outdated years.
3. Call `tavily_extract` ONLY when critical pricing tiers, numbers, or specs are missing from search snippets (hard cap: {MAX_EXTRACT_CALLS} calls, up to {MAX_EXTRACT_URLS} URLs per call).
4. Never invent companies, pricing numbers, metrics, or URLs. If a figure is unavailable, note "Not publicly listed".
5. Every finding MUST be backed by an exact URL returned by `tavily_search` or `tavily_extract`.
6. Once your 2–3 searches (and any needed extractions) have collected solid granular facts across specific companies, product tiers, segments, or regions, immediately write a concise research dossier citing exact URLs for every item."""

STRUCTURER_SYSTEM_PROMPT = """You are a strict data structuring engine.
Convert the provided research dossier into a clean, non-redundant ResearchReport JSON object for an executive Excel spreadsheet.

STRICT RULES:
1. Every item's `source_url` MUST be copied verbatim from the VERIFIED URLS list below. Never invent or alter a URL.
2. Produce 6–10 concrete, granular rows (specific companies, product tiers, business segments, or geographic regions) whenever supported by the research. NEVER create vague umbrella or summary rows such as "Financial Summary", "Company Overview", or "Growth Drivers" — put high-level synthesis in `executive_summary` instead.
3. Keep every field concise, factual, and spreadsheet-ready (no markdown bullets, bold syntax, or raw HTML).
4. `pricing_or_metrics` MUST hold concise numbers, pricing tiers, limits, or percentages ONLY (e.g., "Free: 300 mins; Pro: $16.99/user/mo" or "FY26 Rev: $2.4B; +15% YoY; 22% margin"). No narrative prose. If unavailable, set to "Not publicly listed".
5. `key_details` MUST hold qualitative insights, capabilities, positioning, or growth drivers ONLY — do NOT repeat the exact numbers, dollar figures, or percentages already placed in `pricing_or_metrics`.
6. `notes` MUST provide a brief actionable takeaway, trade-off, or outreach angle without duplicating `key_details` or `pricing_or_metrics`.
7. Deduplicate entities so each row represents one distinct company, product tier, segment, or region."""


@dataclass
class ResearchResult:
    report: ResearchReport
    excel_bytes: bytes
    filename: str
    model_used: str
    verified_url_count: int
    unverified_dropped: int


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p).strip()
    return str(content)


_TRANSIENT_ERROR_MARKERS = (
    "503",
    "504",
    "429",
    "unavailable",
    "gateway timeout",
    "resource_exhausted",
    "resource exhausted",
    "high demand",
    "overloaded",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
)


def _format_exc(exc: BaseException) -> str:
    """Format an exception with class name and repr so timeout/SDK errors never render blank."""
    return f"{type(exc).__name__}: {exc!r}"


def _is_retryable_model_error(exc: Exception) -> bool:
    """Return True when an exception indicates a transient 503/504/429 high-demand or rate-limit error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "code"):
            val = getattr(current, attr, None)
            if val in (429, 503, 504, "429", "503", "504"):
                return True
        msg = str(current).lower()
        if any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _compute_retry_delay() -> float:
    low = max(0.0, MODEL_RETRY_MIN_DELAY_SECONDS)
    high = max(low, MODEL_RETRY_MAX_DELAY_SECONDS)
    return random.uniform(low, high)


def _format_model_label(provider: str, model_name: str) -> str:
    return f"groq/{model_name}" if provider == "groq" else model_name


def _get_retry_attempts(provider: str) -> int:
    return GROQ_RETRY_ATTEMPTS if provider == "groq" else GEMINI_RETRY_ATTEMPTS


def _build_llm(
    provider: str,
    model_name: str,
    gemini_key: str | None,
    groq_key: str | None,
) -> ChatGoogleGenerativeAI | ChatGroq:
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=0,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    if provider == "groq":
        return ChatGroq(
            model=model_name,
            api_key=groq_key,
            temperature=0,
            timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _normalize_search_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _format_research_notes(research_notes: list[str], max_notes_chars: int = 24000) -> str:
    notes_text = "\n\n---\n\n".join(research_notes)
    if len(notes_text) > max_notes_chars:
        notes_text = notes_text[:max_notes_chars] + "\n...[additional cached notes truncated]"
    return notes_text


def _build_stage1_user_prompt(prompt: str, research_notes: list[str]) -> str:
    if not research_notes:
        return prompt
    notes_text = _format_research_notes(research_notes)
    return (
        f"{prompt}\n\n"
        "PREVIOUSLY GATHERED TAVILY RESEARCH FROM THIS RUN "
        "(ALREADY VERIFIED — DO NOT RE-RUN THESE SEARCHES OR RE-EXTRACT THESE URLS):\n"
        f"{notes_text}\n\n"
        "Use the verified findings above first. Only call `tavily_search` or `tavily_extract` "
        "if critical facts or pricing details are still missing."
    )


def _build_dossier_synthesis_prompt(prompt: str, research_notes: list[str]) -> str:
    notes_text = _format_research_notes(research_notes)
    return (
        f"USER REQUEST:\n{prompt}\n\n"
        "VERIFIED TAVILY RESEARCH COLLECTED:\n"
        f"{notes_text}\n\n"
        "Tool budget is complete. Write a concise, source-verified research dossier answering the "
        "user request using ONLY the verified facts and exact URLs above."
    )


def _build_tavily_tools(
    tavily_client: TavilyClient,
    verified_url_map: dict[str, str],
    search_cache: dict[tuple[str, int], str],
    extract_cache: dict[str, str],
    research_notes: list[str],
    tool_usage: dict[str, int],
    log: Callable[[str], None],
) -> list:
    @tool
    def tavily_search(query: str, max_results: int = 5) -> str:
        """Search the live web with Tavily for discovery. Returns titles, snippets, and exact source URLs."""
        cleaned_query = query.strip() if isinstance(query, str) else ""
        if not cleaned_query:
            return (
                "Invalid search query: query cannot be empty or whitespace. "
                "Provide a specific non-empty search query, or synthesize your final dossier now."
            )

        capped = max(1, min(max_results, 5))
        cache_key = (_normalize_search_query(cleaned_query), capped)
        if cache_key in search_cache:
            log(f"Reusing cached Tavily search: `{cleaned_query}`")
            return search_cache[cache_key]

        if tool_usage["search"] >= MAX_SEARCH_CALLS:
            log(f"Search limit reached ({MAX_SEARCH_CALLS}/{MAX_SEARCH_CALLS}) — finalizing dossier...")
            return (
                f"Search limit reached ({MAX_SEARCH_CALLS}/{MAX_SEARCH_CALLS} calls used). "
                "Do NOT call tavily_search again. Write your concise final research dossier now "
                "using the verified facts and URLs you already collected."
            )

        tool_usage["search"] += 1
        try:
            response = tavily_client.search(
                query=cleaned_query,
                max_results=capped,
                search_depth="basic",
                timeout=TAVILY_TIMEOUT_SECONDS,
            )
            results = response.get("results", []) if isinstance(response, dict) else []
            if not results:
                return "No results found for this query."
            blocks = []
            for r in results:
                url = (r.get("url") or "").strip()
                title = (r.get("title") or "Untitled").strip()
                snippet = (r.get("content") or "").strip()
                if url.startswith("http"):
                    verified_url_map[normalize_url(url)] = url
                blocks.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
            formatted = "\n\n---\n\n".join(blocks)
            search_cache[cache_key] = formatted
            research_notes.append(f"[Tavily Search: {cleaned_query}]\n{formatted}")
            return formatted
        except Exception as exc:
            return f"Tavily search error: {exc}"

    @tool
    def tavily_extract(urls: list[str]) -> str:
        """Extract full-page text from up to 4 URLs when pricing tables, specs, or details are missing from snippets."""
        clean_urls = [u.strip() for u in urls if isinstance(u, str) and u.strip().startswith("http")]
        if not clean_urls:
            return "No valid HTTP(S) URLs provided."
        target_urls = clean_urls[:MAX_EXTRACT_URLS]

        uncached_urls: list[str] = []
        seen_norm: set[str] = set()
        for url in target_urls:
            norm = normalize_url(url)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            if norm not in extract_cache:
                uncached_urls.append(url)

        if not uncached_urls:
            log(f"Reusing cached Tavily extraction for {len(seen_norm)} URL(s)...")
            cached_blocks = [
                extract_cache[normalize_url(u)]
                for u in target_urls
                if normalize_url(u) in extract_cache
            ]
            return "\n\n===\n\n".join(dict.fromkeys(cached_blocks))

        if tool_usage["extract"] >= MAX_EXTRACT_CALLS:
            log(f"Extract limit reached ({MAX_EXTRACT_CALLS}/{MAX_EXTRACT_CALLS}) — finalizing dossier...")
            return (
                f"Extract limit reached ({MAX_EXTRACT_CALLS}/{MAX_EXTRACT_CALLS} calls used). "
                "Do NOT call tavily_extract again. Write your concise final research dossier now "
                "using the verified facts and URLs you already collected."
            )

        if len(uncached_urls) < len(seen_norm):
            log(
                f"Reusing {len(seen_norm) - len(uncached_urls)} cached URL(s); "
                f"extracting {len(uncached_urls)} new URL(s)..."
            )

        tool_usage["extract"] += 1
        try:
            response = tavily_client.extract(urls=uncached_urls, timeout=TAVILY_TIMEOUT_SECONDS)
            results = response.get("results", []) if isinstance(response, dict) else []
            new_blocks: list[str] = []
            for r in results:
                url = (r.get("url") or "").strip()
                raw_content = (r.get("raw_content") or "").strip()
                if url.startswith("http"):
                    norm = normalize_url(url)
                    verified_url_map[norm] = url
                    snippet = raw_content[:MAX_EXTRACT_CHARS_PER_URL]
                    if len(raw_content) > MAX_EXTRACT_CHARS_PER_URL:
                        snippet += "\n...[truncated]"
                    block = f"URL: {url}\nExtracted Content:\n{snippet}"
                    extract_cache[norm] = block
                    new_blocks.append(block)
            if new_blocks:
                research_notes.append("[Tavily Extract]\n" + "\n\n===\n\n".join(new_blocks))

            combined_blocks = [
                extract_cache[normalize_url(u)]
                for u in target_urls
                if normalize_url(u) in extract_cache
            ]
            if not combined_blocks:
                return "Could not extract content from the provided URLs."
            return "\n\n===\n\n".join(dict.fromkeys(combined_blocks))
        except Exception as exc:
            return f"Tavily extract error: {exc}"

    return [tavily_search, tavily_extract]


def run_research(
    prompt: str,
    status_callback: Callable[[str], None] | None = None,
) -> ResearchResult:
    """
    Execute a two-stage cloud-only research run:
      1. Tavily Search + Extract driven by LangGraph ReAct agent (Gemini -> Groq fallback chain)
      2. Strict Pydantic structured output synthesis + source URL verification + in-memory .xlsx export
    """
    gemini_key = get_gemini_api_key()
    groq_key = get_groq_api_key()
    tavily_key = get_tavily_api_key()

    if not gemini_key and not groq_key:
        raise RuntimeError(
            "Missing LLM API key. Set GEMINI_API_KEY (and/or GROQ_API_KEY) in your environment variables (or .env file)."
        )
    if not tavily_key:
        raise RuntimeError(
            "Missing TAVILY_API_KEY. Set TAVILY_API_KEY in your environment variables (or .env file)."
        )

    def log(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    active_chain: list[tuple[str, str]] = []
    for provider, model_name in MODEL_CHAIN:
        if provider == "gemini" and gemini_key:
            active_chain.append((provider, model_name))
        elif provider == "groq" and groq_key:
            active_chain.append((provider, model_name))

    if not active_chain:
        raise RuntimeError(
            "No configured models available in MODEL_CHAIN. Check GEMINI_API_KEY / GROQ_API_KEY."
        )

    tavily_client = TavilyClient(api_key=tavily_key)
    verified_url_map: dict[str, str] = {}
    search_cache: dict[tuple[str, int], str] = {}
    extract_cache: dict[str, str] = {}
    research_notes: list[str] = []
    tool_usage: dict[str, int] = {"search": 0, "extract": 0}
    tools = _build_tavily_tools(
        tavily_client=tavily_client,
        verified_url_map=verified_url_map,
        search_cache=search_cache,
        extract_cache=extract_cache,
        research_notes=research_notes,
        tool_usage=tool_usage,
        log=log,
    )

    # Stage 1: Live Web Research with fast cross-provider failover + capped tool loops
    dossier_text = ""
    research_model_spec: tuple[str, str] | None = None
    research_model_used = ""
    stage1_errors: list[str] = []
    stage1_failed_idx: int | None = None

    for model_idx, (provider, model_name) in enumerate(active_chain):
        model_label = _format_model_label(provider, model_name)
        max_attempts = _get_retry_attempts(provider)
        skip_remaining_search = False
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                log(f"Researching with `{model_label}`...")
            else:
                log(
                    f"Retrying research with `{model_label}` "
                    f"(attempt {attempt}/{max_attempts})..."
                )
            attempt_start = time.monotonic()
            try:
                llm = _build_llm(provider, model_name, gemini_key, groq_key)
                react_agent = create_react_agent(llm, tools, prompt=RESEARCH_SYSTEM_PROMPT)
                stage1_prompt = _build_stage1_user_prompt(prompt, research_notes)
                attempt_dossier = ""

                try:
                    for step in react_agent.stream(
                        {"messages": [("user", stage1_prompt)]},
                        config={"recursion_limit": AGENT_RECURSION_LIMIT},
                        stream_mode="values",
                    ):
                        msg = step["messages"][-1]
                        if msg.type == "ai" and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "tool")
                                args = tc.get("args", {})
                                if tool_name == "tavily_search":
                                    q = (args.get("query") or "").strip() if isinstance(args, dict) else ""
                                    if q:
                                        log(f"Searching Tavily: `{q}`")
                                elif tool_name == "tavily_extract":
                                    urls_arg = args.get("urls", []) if isinstance(args, dict) else []
                                    log(f"Extracting full-page details from {len(urls_arg)} URL(s)...")
                                else:
                                    log(f"Calling `{tool_name}`...")
                        elif msg.type == "ai" and msg.content:
                            attempt_dossier = _extract_text(msg.content)
                except GraphRecursionError:
                    if not research_notes:
                        raise
                    log("Research step cap reached — synthesizing concise dossier from collected sources...")
                    synth_msg = llm.invoke(
                        [
                            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                            HumanMessage(content=_build_dossier_synthesis_prompt(prompt, research_notes)),
                        ]
                    )
                    attempt_dossier = _extract_text(synth_msg.content)

                if attempt_dossier.strip():
                    dossier_text = attempt_dossier
                    research_model_spec = (provider, model_name)
                    research_model_used = model_label
                    break
                raise RuntimeError("Model returned an empty research response.")
            except Exception as exc:
                attempt_elapsed = time.monotonic() - attempt_start
                err_detail = _format_exc(exc)
                stage1_errors.append(f"{model_label} (attempt {attempt}): {err_detail}")
                if research_notes:
                    # Earlier searches already succeeded — skip re-running ReAct search and go straight to structuring
                    dossier_text = _format_research_notes(research_notes)
                    stage1_failed_idx = model_idx
                    skip_remaining_search = True
                    log(
                        f"`{model_label}` failed after collecting {len(research_notes)} Tavily result(s) "
                        f"({len(verified_url_map)} verified URLs) — passing findings directly to structuring..."
                    )
                    break
                if (
                    _is_retryable_model_error(exc)
                    and attempt < max_attempts
                    and attempt_elapsed < 10.0
                ):
                    delay = _compute_retry_delay()
                    log(
                        f"`{model_label}` hit a temporary 503/504/429 error "
                        f"(attempt {attempt}/{max_attempts}) — retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                if model_idx < len(active_chain) - 1:
                    log(
                        f"`{model_label}` failed ({type(exc).__name__}) — switching to fallback model..."
                    )
                break

        if dossier_text.strip() or skip_remaining_search:
            break

    if not dossier_text.strip():
        raise RuntimeError(
            "All configured models failed during research. Details: " + " | ".join(stage1_errors)
        )

    # Stage 2: Strict Pydantic Structured Output with fast cross-provider failover
    verified_urls_list = list(verified_url_map.values())
    urls_block = "\n".join(f"- {u}" for u in verified_urls_list) if verified_urls_list else "(none)"

    structuring_prompt = (
        f"USER REQUEST:\n{prompt}\n\n"
        f"VERIFIED URLS RETURNED BY TAVILY:\n{urls_block}\n\n"
        f"RESEARCH DOSSIER:\n{dossier_text}"
    )

    # Choose Stage 2 model order:
    # - If Stage 1 finished with a healthy model, prefer it first.
    # - If Stage 1 short-circuited after collecting Tavily notes on a failing model, start with the NEXT model (e.g. Groq).
    if research_model_spec is not None:
        stage2_models = tuple(dict.fromkeys((research_model_spec, *active_chain)))
    elif stage1_failed_idx is not None:
        rotated = active_chain[stage1_failed_idx + 1 :] + active_chain[: stage1_failed_idx + 1]
        stage2_models = tuple(dict.fromkeys(rotated))
    else:
        stage2_models = tuple(active_chain)

    structured_report: ResearchReport | None = None
    structuring_model_used = ""
    stage2_errors: list[str] = []

    for model_idx, (provider, model_name) in enumerate(stage2_models):
        model_label = _format_model_label(provider, model_name)
        max_attempts = _get_retry_attempts(provider)
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                log(f"Structuring & validating output with `{model_label}`...")
            else:
                log(
                    f"Retrying structuring with `{model_label}` "
                    f"(attempt {attempt}/{max_attempts})..."
                )
            attempt_start = time.monotonic()
            try:
                base_llm = _build_llm(provider, model_name, gemini_key, groq_key)
                messages = [
                    SystemMessage(content=STRUCTURER_SYSTEM_PROMPT),
                    HumanMessage(content=structuring_prompt),
                ]
                try:
                    candidate_report = base_llm.with_structured_output(ResearchReport).invoke(messages)
                except Exception as struct_exc:
                    if provider == "groq" and not _is_retryable_model_error(struct_exc):
                        # Fallback to Groq JSON mode with explicit JSON schema if strict tool-calling fails
                        schema_json = json.dumps(ResearchReport.model_json_schema(), ensure_ascii=False)
                        json_mode_messages = [
                            SystemMessage(
                                content=(
                                    f"{STRUCTURER_SYSTEM_PROMPT}\n\n"
                                    "Return ONLY a single valid JSON object matching this exact JSON Schema:\n"
                                    f"{schema_json}"
                                )
                            ),
                            HumanMessage(content=structuring_prompt),
                        ]
                        candidate_report = base_llm.with_structured_output(
                            ResearchReport,
                            method="json_mode",
                        ).invoke(json_mode_messages)
                    else:
                        raise
                if candidate_report and candidate_report.items:
                    structured_report = candidate_report
                    structuring_model_used = model_label
                    break
                raise RuntimeError("Structured output contained no items.")
            except Exception as exc:
                attempt_elapsed = time.monotonic() - attempt_start
                err_detail = _format_exc(exc)
                stage2_errors.append(f"{model_label} (attempt {attempt}): {err_detail}")
                if (
                    _is_retryable_model_error(exc)
                    and attempt < max_attempts
                    and attempt_elapsed < 10.0
                ):
                    delay = _compute_retry_delay()
                    log(
                        f"Structuring with `{model_label}` hit a temporary 503/504/429 error "
                        f"(attempt {attempt}/{max_attempts}) — retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                if model_idx < len(stage2_models) - 1:
                    log(
                        f"Structuring with `{model_label}` failed ({type(exc).__name__}) — trying fallback model..."
                    )
                break

        if structured_report is not None:
            break

    if structured_report is None:
        raise RuntimeError(
            "All configured models failed while structuring the report. Details: "
            + " | ".join(stage2_errors)
        )

    cleaned_report, unverified_dropped = verify_report_sources(structured_report, verified_url_map)
    excel_bytes = export_report_to_excel_bytes(cleaned_report)
    filename = make_safe_filename(cleaned_report.title)

    # Ensure DataFrame conversion succeeds cleanly before returning
    _ = report_to_dataframe(cleaned_report)

    final_model_used = research_model_used or structuring_model_used

    return ResearchResult(
        report=cleaned_report,
        excel_bytes=excel_bytes,
        filename=filename,
        model_used=final_model_used,
        verified_url_count=len(verified_url_map),
        unverified_dropped=unverified_dropped,
    )
