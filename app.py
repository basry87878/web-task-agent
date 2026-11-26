from __future__ import annotations

import html
import json
import re
import threading
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from agent import ResearchResult, ask_prompt_coach, redact_secrets, run_research, validate_user_input
from config import (
    FALLBACK_MODEL,
    GROQ_FALLBACK_MODEL,
    GROQ_SECONDARY_FALLBACK_MODEL,
    MAX_COACH_MESSAGE_CHARS,
    MAX_COACH_MESSAGES_PER_SESSION,
    MAX_CUSTOM_RUNS_PER_SESSION,
    MAX_RESEARCH_PROMPT_CHARS,
    PRIMARY_MODEL,
    RUN_COOLDOWN_SECONDS,
    get_ga_api_secret,
    get_ga_measurement_id,
    get_gemini_api_key,
    get_groq_api_key,
    get_openrouter_api_key,
    get_tavily_api_key,
)
from schemas import report_to_dataframe, sanitize_excel_cell

LINKEDIN_URL = "https://www.linkedin.com/in/abdelrahmanbasry/"
CONTACT_EMAIL = "a.basry404@gmail.com"

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini-4-hydrogen": "Gemini 4 Hydrogen",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "groq/qwen/qwen3.8-27b": "Qwen 3.8 27B (Groq)",
    "qwen/qwen3.8-27b": "Qwen 3.8 27B (Groq)",
    "groq/openai/gpt-oss-120b": "GPT-OSS 120B (Groq)",
    "openai/gpt-oss-120b": "GPT-OSS 120B (Groq)",
    "groq/openai/gpt-oss-20b": "GPT-OSS 20B (Groq)",
    "openai/gpt-oss-20b": "GPT-OSS 20B (Groq)",
}


def _format_coach_html(text: str) -> str:
    """Safely render assistant/user text inside floating chat bubbles with bold, quotes, and links."""
    escaped = html.escape((text or "").strip())
    lines = escaped.splitlines()
    rendered_lines: list[str] = []
    for line in lines:
        if line.startswith("&gt; "):
            rendered_lines.append(f'<div class="coach-quote">{line[5:]}</div>')
        else:
            rendered_lines.append(line)
    body = "<br>".join(rendered_lines)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(
        r"(https?://[^\s<)]+)",
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        body,
    )
    body = re.sub(
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        r'<a href="mailto:\1">\1</a>',
        body,
    )
    return body


def _format_model_display(raw_model: str) -> str:
    """Map raw provider/model IDs to clean, short human-readable labels."""
    key = (raw_model or "").strip()
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    lower = key.lower()
    if lower in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[lower]
    if key.startswith("groq/"):
        short = key.split("/")[-1]
        return f"{short} (Groq)"
    return key or "Unknown"

st.set_page_config(
    page_title="AI Market Research & Excel Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESETS: list[dict[str, str]] = [
    {
        "id": "b2b_leads",
        "label": "B2B Lead List",
        "badge": "Prospecting",
        "description": "Discover target companies with positioning, size/metrics, outreach angles, and verified URLs.",
        "prompt": (
            "Build a B2B lead list of 6 European B2B SaaS companies specializing in AI workflow automation "
            "or operations automation. For each company, provide their target segment, core offering, "
            "pricing model or team/funding signals, a tailored B2B outreach angle, and their verified official website URL."
        ),
    },
    {
        "id": "competitor_pricing",
        "label": "Competitor Pricing Matrix",
        "badge": "Competitive Intel",
        "description": "Extract live plan tiers, monthly pricing, and limits directly from official pricing pages.",
        "prompt": (
            "Create a competitor pricing matrix comparing 5 leading AI meeting assistant platforms "
            "(Otter.ai, Fireflies.ai, Fathom, tl;dv, and Grain). Use Tavily Extract on their official pricing pages "
            "to capture free vs paid tiers, monthly per-seat pricing, key plan limits, and verified pricing page URLs."
        ),
    },
    {
        "id": "vendor_shortlist",
        "label": "Vendor Shortlist",
        "badge": "Procurement",
        "description": "Compare top B2B vendors on capabilities, pricing model, trade-offs, and best-fit use cases.",
        "prompt": (
            "Shortlist 5 managed cloud vector database vendors suitable for a mid-size engineering team "
            "(such as Pinecone, Qdrant Cloud, Weaviate Cloud, Zilliz Cloud, and Turso/pgvector providers). "
            "Compare deployment model, pricing structure, standout strengths, trade-offs, and verified official URLs."
        ),
    },
]

# Custom CSS for a clean standalone product feel
st.markdown(
    """
    <style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stFooter"] { display: none; }
    [data-testid="stAppDeployButton"] { display: none; }
    [data-testid="stDecoration"] { display: none; }

    .block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1180px; }
    .block-container h1 { padding-top: 0.2rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; line-height: 1.25; }

    .eyebrow-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.24rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(37, 99, 235, 0.25);
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        margin-bottom: 0.25rem;
    }
    .eyebrow-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
    }

    .preset-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.5rem;
        background: rgba(128, 128, 128, 0.04);
    }
    .preset-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.12);
        color: #2563eb;
        margin-bottom: 0.4rem;
    }
    .preset-title { font-size: 1.05rem; font-weight: 700; margin: 0.1rem 0 0.35rem 0; }
    .preset-desc { font-size: 0.88rem; opacity: 0.82; margin: 0; min-height: 2.6rem; }

    .sidebar-contact {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
        background: rgba(128, 128, 128, 0.04);
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    .sidebar-contact-title {
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sidebar-contact a, .cta-callout a, .custom-footer a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }
    .sidebar-contact a:hover, .cta-callout a:hover, .custom-footer a:hover {
        text-decoration: underline;
    }

    .cta-callout {
        margin-top: 0.9rem;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        border: 1px solid rgba(37, 99, 235, 0.28);
        background: rgba(37, 99, 235, 0.06);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .custom-footer {
        margin-top: 2.75rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.8;
    }

    /* Floating Prompt Coach Launcher (stoisser.store dark-navy style) */
    @keyframes coachFloatBob {
        0%, 100% { transform: translateY(0) scale(1.05); }
        50% { transform: translateY(-6px) scale(1.08); }
    }
    .st-key-coach_launcher_wrap {
        position: fixed !important;
        right: 24px !important;
        bottom: 24px !important;
        z-index: 9999 !important;
        width: 56px !important;
        height: 56px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-coach_launcher_wrap button {
        width: 56px !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        border: none !important;
        background: #111827 !important;
        color: #ffffff !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
    }
    .st-key-coach_launcher_wrap button:hover {
        background: #1f2937 !important;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.34) !important;
        animation: coachFloatBob 0.85s ease-in-out infinite !important;
    }
    .st-key-coach_launcher_wrap button:focus,
    .st-key-coach_launcher_wrap button:active {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.26) !important;
        border: none !important;
        color: #ffffff !important;
    }
    .st-key-coach_launcher_wrap button span,
    .st-key-coach_launcher_wrap button p {
        color: #ffffff !important;
        font-size: 23px !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    /* Floating Chat Card anchored above launcher */
    .st-key-coach_panel_card {
        position: fixed !important;
        right: 24px !important;
        bottom: 92px !important;
        z-index: 9998 !important;
        width: 370px !important;
        max-width: calc(100vw - 32px) !important;
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18) !important;
        overflow: hidden !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    .st-key-coach_panel_card > div {
        gap: 0 !important;
    }

    /* Dark-navy card header */
    .st-key-coach_header_bar {
        background: #111827 !important;
        color: #ffffff !important;
        padding: 12px 14px 12px 16px !important;
        width: 100% !important;
        margin: 0 !important;
        align-items: center !important;
    }
    .st-key-coach_header_bar > div {
        display: flex !important;
        align-items: center !important;
    }
    .coach-hdr-title {
        font-weight: 600;
        font-size: 15px;
        color: #ffffff;
        line-height: 1.25;
    }
    .coach-hdr-sub {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 2px;
        line-height: 1.25;
    }
    .st-key-coach_close_btn {
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    .st-key-coach_close_btn button {
        background: transparent !important;
        border: none !important;
        color: #d1d5db !important;
        font-size: 20px !important;
        line-height: 1 !important;
        padding: 4px 8px !important;
        min-height: 28px !important;
        box-shadow: none !important;
    }
    .st-key-coach_close_btn button:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* Scrollable messages area */
    .st-key-coach_messages_scroll {
        background: #f9fafb !important;
        padding: 12px 14px !important;
        gap: 8px !important;
    }
    .coach-msg-row {
        display: flex;
        width: 100%;
        margin: 0;
    }
    .coach-msg-row.bot { justify-content: flex-start; }
    .coach-msg-row.user { justify-content: flex-end; }
    .coach-msg {
        max-width: 88%;
        padding: 10px 13px;
        border-radius: 12px;
        font-size: 13.5px;
        line-height: 1.45;
        word-wrap: break-word;
    }
    .coach-msg.bot {
        background: #ffffff;
        color: #111827;
        border: 1px solid #e5e7eb;
    }
    .coach-msg.user {
        background: #111827;
        color: #ffffff;
    }
    .coach-msg a {
        color: #2563eb;
        text-decoration: underline;
    }
    .coach-msg.user a { color: #93c5fd; }
    .coach-quote {
        margin: 6px 0;
        padding: 7px 10px;
        border-left: 3px solid #111827;
        background: #f3f4f6;
        border-radius: 6px;
        font-size: 12.5px;
        color: #111827;
    }

    /* 1-click "Use this prompt" action chip */
    [class*="st-key-use_coach_prompt_"] button {
        border-radius: 999px !important;
        border: 1px solid #bfdbfe !important;
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 4px 12px !important;
        min-height: 30px !important;
    }
    [class*="st-key-use_coach_prompt_"] button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
    }

    /* 2 compact pill starter chips above input */
    .st-key-coach_chips_row {
        background: #f9fafb !important;
        border-top: 1px solid #f3f4f6 !important;
        padding: 8px 12px !important;
        width: 100% !important;
    }
    .st-key-coach_chips_row button {
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        background: #ffffff !important;
        color: #111827 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 5px 11px !important;
        min-height: 30px !important;
        transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
    }
    .st-key-coach_chips_row button:hover {
        background: #111827 !important;
        color: #ffffff !important;
        border-color: #111827 !important;
    }
    .st-key-coach_chips_row button p {
        font-size: 12px !important;
        margin: 0 !important;
    }

    /* Bottom input row & dark-navy Send button */
    .st-key-coach_input_bar {
        background: #ffffff !important;
        border-top: 1px solid #e5e7eb !important;
        padding: 10px 10px 4px 10px !important;
        width: 100% !important;
    }
    .st-key-coach_input_bar [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }
    .st-key-coach_input_bar input {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        padding: 8px 11px !important;
    }
    .st-key-coach_input_bar input:focus {
        border-color: #111827 !important;
        box-shadow: none !important;
    }
    .st-key-coach_input_bar [data-testid="stFormSubmitButton"] button {
        background: #111827 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        min-height: 38px !important;
        padding: 0 12px !important;
    }
    .st-key-coach_input_bar [data-testid="stFormSubmitButton"] button:hover {
        background: #1f2937 !important;
        color: #ffffff !important;
    }
    .st-key-coach_input_bar [data-testid="stFormSubmitButton"] button:disabled {
        opacity: 0.45 !important;
    }

    .st-key-coach_footer_bar {
        background: #ffffff !important;
        padding: 2px 12px 10px 12px !important;
        width: 100% !important;
        margin: 0 !important;
    }
    .st-key-coach_footer_bar [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    .coach-card-footer {
        font-size: 11px;
        line-height: 1.4;
        color: #6b7280;
        text-align: center;
        margin: 0;
        padding: 0;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalize_prompt_cache_key(prompt: str) -> str:
    """Normalize case and whitespace so identical custom prompts reuse cached results."""
    return " ".join((prompt or "").strip().lower().split())


@st.cache_data(ttl=3600, show_spinner=False)
def _run_cached_preset(preset_id: str, prompt: str) -> ResearchResult:
    """Cache preset runs for 1 hour to protect API quotas across public visitors."""
    return run_research(prompt)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_cached_custom(normalized_key: str, _raw_prompt: str) -> ResearchResult:
    """Cache normalized custom prompts for 1 hour so identical queries return instantly without extra Tavily calls."""
    return run_research(_raw_prompt)


def _inject_ga4_gtag() -> None:
    """Inject GA4 gtag.js into parent <head> once per session (silently no-ops if unconfigured)."""
    measurement_id = get_ga_measurement_id()
    if not measurement_id or st.session_state.get("ga4_injected"):
        return
    st.session_state["ga4_injected"] = True
    mid_js = json.dumps(measurement_id)
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                var doc = window.parent.document;
                if (doc.getElementById("ga4-gtag-script")) return;
                var mid = {mid_js};
                var s1 = doc.createElement("script");
                s1.id = "ga4-gtag-script";
                s1.async = true;
                s1.src = "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(mid);
                doc.head.appendChild(s1);

                var s2 = doc.createElement("script");
                s2.id = "ga4-gtag-config";
                s2.text =
                    "window.dataLayer = window.dataLayer || [];" +
                    "function gtag(){{dataLayer.push(arguments);}}" +
                    "gtag('js', new Date());" +
                    "gtag('config', " + JSON.stringify(mid) + ");";
                doc.head.appendChild(s2);
            }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def track_ga_event(
    event_name: str,
    params: dict[str, str | int | float | bool] | None = None,
) -> None:
    """
    Fire-and-forget GA4 Measurement Protocol event in a background daemon thread.
    Silently no-ops if GA_MEASUREMENT_ID or GA_API_SECRET is unset, and never blocks or raises.
    """
    measurement_id = get_ga_measurement_id()
    api_secret = get_ga_api_secret()
    if not measurement_id or not api_secret:
        return

    client_id = st.session_state.get("ga_client_id") or str(uuid.uuid4())
    safe_params: dict[str, str | int | float | bool] = {"engagement_time_msec": 100}
    if params:
        for k, v in params.items():
            if isinstance(v, bool):
                safe_params[k] = v
            elif isinstance(v, (int, float)):
                safe_params[k] = v
            elif isinstance(v, str):
                safe_params[k] = redact_secrets(v)[:100]

    payload = {
        "client_id": client_id,
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


# Initialize per-visitor session state
if "ga_client_id" not in st.session_state:
    st.session_state["ga_client_id"] = str(uuid.uuid4())
if "ga4_injected" not in st.session_state:
    st.session_state["ga4_injected"] = False
if "custom_runs_used" not in st.session_state:
    st.session_state["custom_runs_used"] = 0
if "last_run_completed_at" not in st.session_state:
    st.session_state["last_run_completed_at"] = 0.0
if "history" not in st.session_state:
    st.session_state["history"] = []  # list[dict]
if "active_result_idx" not in st.session_state:
    st.session_state["active_result_idx"] = None
if "custom_prompt_input" not in st.session_state:
    st.session_state["custom_prompt_input"] = ""
if "coach_messages" not in st.session_state:
    st.session_state["coach_messages"] = []  # list[dict]
if "coach_assistant_turns" not in st.session_state:
    st.session_state["coach_assistant_turns"] = 0
if "coach_prompt_applied" not in st.session_state:
    st.session_state["coach_prompt_applied"] = False
if "coach_open" not in st.session_state:
    st.session_state["coach_open"] = False
if "coach_error" not in st.session_state:
    st.session_state["coach_error"] = None

_inject_ga4_gtag()


def _toggle_coach() -> None:
    st.session_state["coach_open"] = not st.session_state["coach_open"]


def _close_coach() -> None:
    st.session_state["coach_open"] = False


def _apply_suggested_prompt(prompt_text: str) -> None:
    """Populate the main Custom Research text area from a 1-click Prompt Coach suggestion."""
    st.session_state["custom_prompt_input"] = prompt_text
    st.session_state["coach_prompt_applied"] = True
    st.session_state["coach_open"] = False


# Sidebar: session-only history, contact block, and collapsed stack info
with st.sidebar:
    st.header("Recent Exports")
    if st.session_state["history"]:
        for idx, entry in enumerate(reversed(st.session_state["history"])):
            real_idx = len(st.session_state["history"]) - 1 - idx
            res: ResearchResult = entry["result"]
            if st.button(
                f"📄 {res.report.title[:34]}",
                key=f"hist_btn_{real_idx}",
                use_container_width=True,
            ):
                st.session_state["active_result_idx"] = real_idx
    else:
        st.info("Run a preset or custom prompt to generate a source-verified Excel report.")

    st.divider()
    st.markdown(
        f"""
        <div class="sidebar-contact">
            <div class="sidebar-contact-title">Built by Abdelrahman Basry</div>
            <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">LinkedIn</a>
            &nbsp;·&nbsp;
            <a href="mailto:{CONTACT_EMAIL}">Email</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Under the Hood", expanded=False):
        engine_chain = " → ".join(
            _format_model_display(m)
            for m in (
                PRIMARY_MODEL or "",
                FALLBACK_MODEL or "",
                f"groq/{GROQ_FALLBACK_MODEL}" if GROQ_FALLBACK_MODEL else "",
                f"groq/{GROQ_SECONDARY_FALLBACK_MODEL}" if GROQ_SECONDARY_FALLBACK_MODEL else "",
            )
            if m
        )
        st.caption(
            f"**Engine:** {engine_chain}  \n"
            f"**Search:** Tavily Search + Extract  \n"
            f"**Export:** Strict Pydantic → In-memory `.xlsx`"
        )


# Main header
st.markdown(
    '<div class="eyebrow-pill"><span class="eyebrow-dot"></span>'
    "LIVE AI AGENT · ~15s PER REPORT · VERIFIED SOURCES</div>",
    unsafe_allow_html=True,
)
st.title("📊 AI Market Research & Excel Builder")
st.markdown(
    "Turn any B2B research brief into a **source-verified, spreadsheet-ready `.xlsx` report** "
    "using live Tavily search/extraction and Gemini structured outputs."
)

# API key readiness banner (never reveals keys)
missing_keys = []
if not get_gemini_api_key() and not get_groq_api_key():
    missing_keys.append("`GEMINI_API_KEY` (or `GROQ_API_KEY`)")
if not get_tavily_api_key():
    missing_keys.append("`TAVILY_API_KEY`")
if missing_keys:
    st.warning(
        f"Missing environment variable(s): {', '.join(missing_keys)}. "
        "Add them to `.env` locally or in your Render Environment Variables to run live research."
    )

# Section 1: 3 Clickable Business Presets
st.subheader("1. Try a Business Preset")
preset_cols = st.columns(3, gap="medium")
triggered_preset: dict[str, str] | None = None

for col, preset in zip(preset_cols, PRESETS):
    with col:
        st.markdown(
            f"""
            <div class="preset-card">
                <span class="preset-badge">{preset['badge']}</span>
                <div class="preset-title">{preset['label']}</div>
                <p class="preset-desc">{preset['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Run {preset['label']}",
            key=f"preset_{preset['id']}",
            use_container_width=True,
            disabled=bool(missing_keys),
        ):
            triggered_preset = preset

st.divider()

# Section 2: Custom Research Prompt
st.subheader("2. Or Run a Custom Research Brief")

if st.session_state.get("coach_prompt_applied"):
    st.success("Loaded the improved prompt into the Custom Research Brief box below.")
    st.session_state["coach_prompt_applied"] = False

session_limit_reached = st.session_state["custom_runs_used"] >= MAX_CUSTOM_RUNS_PER_SESSION

with st.form("custom_research_form", clear_on_submit=False):
    custom_prompt = st.text_area(
        "Describe what you want researched",
        key="custom_prompt_input",
        max_chars=MAX_RESEARCH_PROMPT_CHARS,
        placeholder=(
            "e.g., Compare pricing and key features of 5 European payroll API providers "
            "for startups, including their official pricing page URLs."
        ),
        height=100,
        disabled=(session_limit_reached or bool(missing_keys)),
    )
    col_submit, _ = st.columns([1, 2])
    with col_submit:
        submit_custom = st.form_submit_button(
            "Run Custom Research",
            type="primary",
            use_container_width=True,
            disabled=(session_limit_reached or bool(missing_keys)),
        )

if session_limit_reached:
    st.info(
        f"You've reached this session's safety limit of {MAX_CUSTOM_RUNS_PER_SESSION} custom runs. "
        "You can still explore the presets or download your generated `.xlsx` reports below."
    )


def _check_run_cooldown() -> int:
    """Return remaining cooldown seconds (0 if ready to run)."""
    last_ts = st.session_state.get("last_run_completed_at", 0.0)
    if last_ts <= 0:
        return 0
    elapsed = time.monotonic() - last_ts
    if elapsed < RUN_COOLDOWN_SECONDS:
        return max(1, int(RUN_COOLDOWN_SECONDS - elapsed + 0.999))
    return 0


def _execute_and_store(prompt: str, is_preset: bool, preset_id: str | None = None) -> None:
    wait_secs = _check_run_cooldown()
    if wait_secs > 0:
        st.info(f"Please wait {wait_secs}s before starting another research run.")
        return

    with st.status("Running live Tavily research & structuring report...", expanded=True) as status:
        t_start = time.perf_counter()
        try:
            if is_preset and preset_id:
                st.write("Checking 1-hour cache / running Tavily + AI pipeline...")
                result = _run_cached_preset(preset_id, prompt)
            else:
                st.write("Checking 1-hour cache / running live Tavily + AI pipeline...")
                norm_key = _normalize_prompt_cache_key(prompt)
                result = _run_cached_custom(norm_key, prompt)

            elapsed_seconds = time.perf_counter() - t_start
            if not is_preset and elapsed_seconds >= 0.5:
                # Only count live uncached runs toward the quiet background safety guardrail
                st.session_state["custom_runs_used"] += 1
            st.session_state["last_run_completed_at"] = time.monotonic()

            st.session_state["history"].append(
                {
                    "prompt": prompt,
                    "is_preset": is_preset,
                    "result": result,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            st.session_state["active_result_idx"] = len(st.session_state["history"]) - 1
            model_label = _format_model_display(result.model_used)
            track_ga_event(
                "preset_run" if is_preset else "custom_run",
                {
                    "preset_id": preset_id or "custom",
                    "model_used": model_label,
                    "rows_count": len(result.report.items),
                    "verified_urls": result.verified_url_count,
                    "duration_sec": round(elapsed_seconds, 2),
                    "cached": elapsed_seconds < 0.5,
                },
            )
            status.update(
                label=(
                    f"Completed with {model_label} in {elapsed_seconds:.1f}s — "
                    f"{len(result.report.items)} verified rows ready"
                ),
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            st.session_state["last_run_completed_at"] = time.monotonic()
            status.update(label="Research run failed", state="error", expanded=True)
            st.error(f"Could not complete research: {redact_secrets(str(exc))}")


if triggered_preset is not None:
    _execute_and_store(
        prompt=triggered_preset["prompt"],
        is_preset=True,
        preset_id=triggered_preset["id"],
    )
elif submit_custom:
    is_valid, validated_or_err = validate_user_input(
        custom_prompt or "",
        max_chars=MAX_RESEARCH_PROMPT_CHARS,
        min_chars=10,
    )
    if not is_valid:
        st.warning(validated_or_err)
    elif st.session_state["custom_runs_used"] >= MAX_CUSTOM_RUNS_PER_SESSION:
        st.info(
            f"You've reached this session's safety limit of {MAX_CUSTOM_RUNS_PER_SESSION} custom runs. "
            "You can still explore the presets or download your generated `.xlsx` reports below."
        )
    else:
        _execute_and_store(prompt=validated_or_err, is_preset=False)


# Section 3: Inline Preview Table + Download .xlsx Button
active_idx = st.session_state.get("active_result_idx")
if active_idx is not None and 0 <= active_idx < len(st.session_state["history"]):
    entry = st.session_state["history"][active_idx]
    res: ResearchResult = entry["result"]
    elapsed_seconds = entry.get("elapsed_seconds")
    df = report_to_dataframe(res.report)

    st.divider()
    st.subheader(res.report.title)
    st.write(res.report.executive_summary)

    m1, m2, m3, m4 = st.columns([1, 1, 1.9, 1])
    m1.metric("Structured Rows", len(df))
    m2.metric("Verified Tavily URLs", res.verified_url_count)
    m3.metric("Model Used", _format_model_display(res.model_used))
    m4.metric("Run Time", f"{elapsed_seconds:.1f}s" if elapsed_seconds is not None else "—")

    if res.unverified_dropped > 0:
        st.caption(
            f"Guardrail note: {res.unverified_dropped} row(s) with unverified links were filtered out automatically."
        )

    # Inline interactive preview table BEFORE the download buttons
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Key Details": st.column_config.TextColumn("Key Details", width="large"),
            "Pricing / Metrics": st.column_config.TextColumn("Pricing / Metrics", width="medium"),
            "Notes": st.column_config.TextColumn("Notes", width="medium"),
            "Source URL": st.column_config.LinkColumn(
                "Source URL",
                display_text="Open source",
                width="small",
            ),
        },
    )

    safe_csv_df = df.copy()
    for col in safe_csv_df.columns:
        safe_csv_df[col] = safe_csv_df[col].map(sanitize_excel_cell)
    csv_bytes = safe_csv_df.to_csv(index=False).encode("utf-8-sig")
    csv_filename = (
        res.filename[:-5] + ".csv" if res.filename.endswith(".xlsx") else f"{res.filename}.csv"
    )

    dl_excel_col, dl_csv_col = st.columns([2.3, 1], gap="small")
    with dl_excel_col:
        if st.download_button(
            label=f"⬇️ Download Excel Report ({res.filename})",
            data=res.excel_bytes,
            file_name=res.filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        ):
            track_ga_event(
                "excel_download",
                {
                    "model_used": _format_model_display(res.model_used),
                    "rows_count": len(df),
                    "verified_urls": res.verified_url_count,
                },
            )
    with dl_csv_col:
        if st.download_button(
            label="📄 Download CSV",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            type="secondary",
            use_container_width=True,
        ):
            track_ga_event(
                "csv_download",
                {
                    "model_used": _format_model_display(res.model_used),
                    "rows_count": len(df),
                    "verified_urls": res.verified_url_count,
                },
            )

    st.markdown(
        f"""
        <div class="cta-callout">
            Want a custom version connected to your CRM, Power BI, or internal data?
            <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">Connect on LinkedIn</a>
            or email
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="custom-footer">
        © 2026 Basry. All rights reserved. ·
        <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">Built by Basry</a> ·
        <a href="mailto:{CONTACT_EMAIL}">Email</a>
    </div>
    """,
    unsafe_allow_html=True,
)


# Floating Bottom-Right Prompt Coach Widget (stoisser.store style)
def _render_floating_coach() -> None:
    has_openrouter = bool(get_openrouter_api_key())
    coach_remaining = max(
        0, MAX_COACH_MESSAGES_PER_SESSION - st.session_state["coach_assistant_turns"]
    )
    coach_disabled = (not has_openrouter) or (coach_remaining == 0)

    if st.session_state["coach_open"]:
        with st.container(key="coach_panel_card"):
            # 1. Dark-navy header bar
            with st.container(
                key="coach_header_bar",
                horizontal=True,
                vertical_alignment="center",
                horizontal_alignment="distribute",
            ):
                st.markdown(
                    """
                    <div>
                        <div class="coach-hdr-title">Research Assistant</div>
                        <div class="coach-hdr-sub">Prompt Coach · Tool Guide</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button("×", key="coach_close_btn", on_click=_close_coach)

            # 2. Scrollable chat messages area
            with st.container(
                key="coach_messages_scroll",
                height=240,
                border=False,
                autoscroll=True,
            ):
                st.markdown(
                    """
                    <div class="coach-msg-row bot">
                        <div class="coach-msg bot">
                            Hi! 👋 Paste a rough research idea and I'll sharpen it into a 2026 brief, or ask how this tool works.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not has_openrouter:
                    st.markdown(
                        """
                        <div class="coach-msg-row bot">
                            <div class="coach-msg bot">
                                Prompt Coach needs <strong>OPENROUTER_API_KEY</strong> in environment variables. The main research tool still works normally.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                for idx, msg in enumerate(st.session_state["coach_messages"]):
                    role_cls = "user" if msg["role"] == "user" else "bot"
                    st.markdown(
                        f'<div class="coach-msg-row {role_cls}">'
                        f'<div class="coach-msg {role_cls}">{_format_coach_html(msg["content"])}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if msg["role"] == "assistant" and msg.get("suggested_prompt"):
                        st.button(
                            "✨ Use this prompt",
                            key=f"use_coach_prompt_{idx}",
                            on_click=_apply_suggested_prompt,
                            args=(msg["suggested_prompt"],),
                        )

                if st.session_state.get("coach_error"):
                    st.markdown(
                        f'<div class="coach-msg-row bot">'
                        f'<div class="coach-msg bot">{html.escape(st.session_state["coach_error"])}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # 3. Two compact pill starter chips right above the input row
            starter_q: str | None = None
            with st.container(key="coach_chips_row", horizontal=True, gap="small"):
                if st.button(
                    "Improve my prompt",
                    key="coach_chip_improve",
                    disabled=coach_disabled,
                ):
                    draft = (st.session_state.get("custom_prompt_input") or "").strip()
                    if draft:
                        draft_ok, draft_val = validate_user_input(
                            draft,
                            max_chars=MAX_RESEARCH_PROMPT_CHARS,
                            min_chars=1,
                        )
                        if not draft_ok:
                            st.session_state["coach_error"] = draft_val
                            st.rerun()
                        prefix = "Improve this 2026 B2B research prompt: "
                        max_draft = max(50, MAX_COACH_MESSAGE_CHARS - len(prefix))
                        starter_q = prefix + draft_val[:max_draft]
                    else:
                        starter_q = (
                            "Improve my prompt — share one sharp, ready-to-run 2026 B2B research "
                            "brief example and ask what topic or industry I want to research."
                        )
                if st.button(
                    "How does this work?",
                    key="coach_chip_how",
                    disabled=coach_disabled,
                ):
                    starter_q = "How does this work?"

            # 4. Input box + dark-navy Send button at the bottom of the card
            with st.container(key="coach_input_bar"):
                with st.form("prompt_coach_form", clear_on_submit=True, border=False):
                    in_col, send_col = st.columns([3.2, 1.1], gap="small", vertical_alignment="center")
                    with in_col:
                        typed_q = st.text_input(
                            "Message",
                            max_chars=MAX_COACH_MESSAGE_CHARS,
                            placeholder="Write a message...",
                            disabled=coach_disabled,
                            label_visibility="collapsed",
                        )
                    with send_col:
                        submitted_q = st.form_submit_button(
                            "Send",
                            use_container_width=True,
                            disabled=coach_disabled,
                        )

            # 5. Subtle footer line
            with st.container(key="coach_footer_bar"):
                st.markdown(
                    f'<div class="coach-card-footer">{coach_remaining}/{MAX_COACH_MESSAGES_PER_SESSION} '
                    "replies left · Powered by OpenRouter</div>",
                    unsafe_allow_html=True,
                )

            # Process chip click or form submission
            pending_q = starter_q or (typed_q.strip() if submitted_q else "")
            if pending_q and not coach_disabled:
                is_valid_q, validated_q_or_err = validate_user_input(
                    pending_q,
                    max_chars=MAX_COACH_MESSAGE_CHARS,
                    min_chars=1,
                )
                if not is_valid_q:
                    st.session_state["coach_error"] = validated_q_or_err
                    st.rerun()
                st.session_state["coach_error"] = None
                st.session_state["coach_messages"].append(
                    {"role": "user", "content": validated_q_or_err, "suggested_prompt": None}
                )
                with st.spinner("Thinking..."):
                    t_coach = time.perf_counter()
                    try:
                        reply_md, suggested = ask_prompt_coach(st.session_state["coach_messages"])
                        coach_elapsed = time.perf_counter() - t_coach
                        st.session_state["coach_messages"].append(
                            {
                                "role": "assistant",
                                "content": reply_md,
                                "suggested_prompt": suggested,
                            }
                        )
                        st.session_state["coach_assistant_turns"] += 1
                        track_ga_event(
                            "prompt_coach_message",
                            {
                                "turn_number": st.session_state["coach_assistant_turns"],
                                "has_suggested_prompt": bool(suggested),
                                "duration_sec": round(coach_elapsed, 2),
                            },
                        )
                    except Exception as exc:
                        st.session_state["coach_messages"].pop()
                        st.session_state["coach_error"] = redact_secrets(str(exc))
                st.rerun()

    # Fixed bottom-right circular dark-navy launcher button
    with st.container(key="coach_launcher_wrap"):
        st.button(
            "",
            key="coach_launcher_btn",
            icon=":material/chat_bubble:",
            on_click=_toggle_coach,
        )


_render_floating_coach()
