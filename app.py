from __future__ import annotations

import time

import streamlit as st

from agent import ResearchResult, run_research
from config import (
    FALLBACK_MODEL,
    GROQ_FALLBACK_MODEL,
    GROQ_SECONDARY_FALLBACK_MODEL,
    MAX_CUSTOM_RUNS_PER_SESSION,
    PRIMARY_MODEL,
    get_gemini_api_key,
    get_groq_api_key,
    get_tavily_api_key,
)
from schemas import report_to_dataframe

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

    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1180px; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; line-height: 1.25; }

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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_cached_preset(preset_id: str, prompt: str) -> ResearchResult:
    """Cache preset demos for 1 hour to protect free API quotas across public visitors."""
    return run_research(prompt)


# Initialize per-visitor session state
if "custom_runs_used" not in st.session_state:
    st.session_state["custom_runs_used"] = 0
if "history" not in st.session_state:
    st.session_state["history"] = []  # list[dict]
if "active_result_idx" not in st.session_state:
    st.session_state["active_result_idx"] = None


# Sidebar: quota indicator, configuration status, and session-only history
with st.sidebar:
    st.header("Session Quota")
    used = st.session_state["custom_runs_used"]
    remaining = max(0, MAX_CUSTOM_RUNS_PER_SESSION - used)
    st.progress(min(1.0, used / MAX_CUSTOM_RUNS_PER_SESSION))
    st.caption(
        f"**{remaining} of {MAX_CUSTOM_RUNS_PER_SESSION}** custom research runs remaining this session. "
        "Business presets are cached and do not use your custom quota."
    )

    st.divider()
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

# Section 2: Custom Research Prompt with Session Rate Limit
st.subheader("2. Or Run a Custom Research Brief")
custom_remaining = max(0, MAX_CUSTOM_RUNS_PER_SESSION - st.session_state["custom_runs_used"])

with st.form("custom_research_form", clear_on_submit=False):
    custom_prompt = st.text_area(
        "Describe what you want researched",
        placeholder=(
            "e.g., Compare pricing and key features of 5 European payroll API providers "
            "for startups, including their official pricing page URLs."
        ),
        height=100,
        disabled=(custom_remaining == 0 or bool(missing_keys)),
    )
    col_submit, col_quota = st.columns([1, 2])
    with col_submit:
        submit_custom = st.form_submit_button(
            "Run Custom Research",
            type="primary",
            use_container_width=True,
            disabled=(custom_remaining == 0 or bool(missing_keys)),
        )
    with col_quota:
        if custom_remaining > 0:
            st.caption(
                f"Free demo limit: **{custom_remaining} custom run(s)** left in this session."
            )
        else:
            st.error(
                f"You've reached the session limit ({MAX_CUSTOM_RUNS_PER_SESSION} custom runs). "
                "You can still explore the presets or download your generated `.xlsx` reports below."
            )


def _execute_and_store(prompt: str, is_preset: bool, preset_id: str | None = None) -> None:
    with st.status("Running live Tavily research & structuring report...", expanded=True) as status:
        t_start = time.perf_counter()
        try:
            if is_preset and preset_id:
                st.write("Checking 1-hour preset cache / running Tavily + AI pipeline...")
                result = _run_cached_preset(preset_id, prompt)
            else:
                result = run_research(prompt, status_callback=st.write)
                st.session_state["custom_runs_used"] += 1

            elapsed_seconds = time.perf_counter() - t_start
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
            status.update(
                label=(
                    f"Completed with {model_label} in {elapsed_seconds:.1f}s — "
                    f"{len(result.report.items)} verified rows ready"
                ),
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            status.update(label="Research run failed", state="error", expanded=True)
            st.error(f"Could not complete research: {exc}")


if triggered_preset is not None:
    _execute_and_store(
        prompt=triggered_preset["prompt"],
        is_preset=True,
        preset_id=triggered_preset["id"],
    )
elif submit_custom:
    cleaned_prompt = (custom_prompt or "").strip()
    if len(cleaned_prompt) < 10:
        st.warning("Please enter a slightly more detailed research prompt (at least 10 characters).")
    elif st.session_state["custom_runs_used"] >= MAX_CUSTOM_RUNS_PER_SESSION:
        st.error(f"Session limit of {MAX_CUSTOM_RUNS_PER_SESSION} custom runs reached.")
    else:
        _execute_and_store(prompt=cleaned_prompt, is_preset=False)


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

    # Inline interactive preview table BEFORE the download button
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source URL": st.column_config.LinkColumn(
                "Source URL",
                display_text="Open source",
            ),
        },
    )

    st.download_button(
        label=f"⬇️ Download Excel Report ({res.filename})",
        data=res.excel_bytes,
        file_name=res.filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
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
