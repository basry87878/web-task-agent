from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env first, then fall back to repo root .env for seamless local development
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
load_dotenv(_BACKEND_DIR / ".env", override=False)
load_dotenv(_REPO_ROOT / ".env", override=False)


def _get_env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val and val.strip():
        return val.strip()
    return default


def get_gemini_api_key() -> str | None:
    """Return GEMINI_API_KEY (falling back to GOOGLE_API_KEY for local compatibility)."""
    return _get_env("GEMINI_API_KEY") or _get_env("GOOGLE_API_KEY")


def get_tavily_api_key() -> str | None:
    """Return TAVILY_API_KEY."""
    return _get_env("TAVILY_API_KEY")


def get_groq_api_key() -> str | None:
    """Return GROQ_API_KEY for cross-provider failover and fast Prompt Coach replies."""
    return _get_env("GROQ_API_KEY")


def get_openrouter_api_key() -> str | None:
    """Return OPENROUTER_API_KEY (fallback provider for Prompt Coach)."""
    return _get_env("OPENROUTER_API_KEY")


def get_ga_measurement_id() -> str | None:
    """Return optional Google Analytics 4 Measurement ID (e.g., G-XXXXXXXXXX)."""
    return _get_env("GA_MEASUREMENT_ID")


def get_ga_api_secret() -> str | None:
    """Return optional Google Analytics 4 Measurement Protocol API secret."""
    return _get_env("GA_API_SECRET")


# Cloud-only cross-provider fallback chain:
#   gemini-4-hydrogen (1 fast attempt) -> gemini-3.5-flash-lite (1 fast attempt)
#   -> Groq qwen/qwen3.8-27b -> Groq openai/gpt-oss-120b
PRIMARY_MODEL = _get_env("GEMINI_PRIMARY_MODEL", "gemini-4-hydrogen")
FALLBACK_MODEL = _get_env("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
GROQ_FALLBACK_MODEL = _get_env("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_SECONDARY_FALLBACK_MODEL = _get_env("GROQ_SECONDARY_FALLBACK_MODEL", "openai/gpt-oss-120b")


def _build_model_chain() -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[str, str]] = [
        ("gemini", PRIMARY_MODEL or ""),
        ("gemini", FALLBACK_MODEL or ""),
        ("groq", GROQ_FALLBACK_MODEL or ""),
        ("groq", GROQ_SECONDARY_FALLBACK_MODEL or ""),
    ]
    return tuple(
        dict.fromkeys(
            (provider, model.strip())
            for provider, model in candidates
            if model and model.strip()
        )
    )


MODEL_CHAIN: tuple[tuple[str, str], ...] = _build_model_chain()

# Fast failover timeouts & retry counts
LLM_REQUEST_TIMEOUT_SECONDS = float(_get_env("LLM_REQUEST_TIMEOUT_SECONDS", "22") or "22")
GROQ_REQUEST_TIMEOUT_SECONDS = float(_get_env("GROQ_REQUEST_TIMEOUT_SECONDS", "40") or "40")
LLM_MAX_RETRIES = 1  # Disables hidden multi-minute google-genai SDK backoff (1 attempt per request)
GEMINI_RETRY_ATTEMPTS = max(1, int(_get_env("GEMINI_RETRY_ATTEMPTS", "1") or "1"))
GROQ_RETRY_ATTEMPTS = max(1, int(_get_env("GROQ_RETRY_ATTEMPTS", "1") or "1"))
MODEL_RETRY_MIN_DELAY_SECONDS = float(_get_env("MODEL_RETRY_MIN_DELAY_SECONDS", "1.5") or "1.5")
MODEL_RETRY_MAX_DELAY_SECONDS = float(_get_env("MODEL_RETRY_MAX_DELAY_SECONDS", "2.5") or "2.5")
TAVILY_TIMEOUT_SECONDS = float(_get_env("TAVILY_TIMEOUT_SECONDS", "15") or "15")

# Prompt Coach settings (Groq -> Gemini -> OpenRouter fallback)
OPENROUTER_MODEL = _get_env("OPENROUTER_MODEL", "openrouter/free") or "openrouter/free"
OPENROUTER_TIMEOUT_SECONDS = float(_get_env("OPENROUTER_TIMEOUT_SECONDS", "20") or "20")
COACH_GROQ_TIMEOUT_SECONDS = float(_get_env("COACH_GROQ_TIMEOUT_SECONDS", "18") or "18")

# Server-side rate limiting & caching guardrails
MAX_RESEARCH_RUNS_PER_SESSION = int(_get_env("MAX_RESEARCH_RUNS_PER_SESSION", "3") or "3")
MAX_COACH_REPLIES_PER_SESSION = int(_get_env("MAX_COACH_REPLIES_PER_SESSION", "10") or "10")
RATE_LIMIT_TTL_SECONDS = int(_get_env("RATE_LIMIT_TTL_SECONDS", "3600") or "3600")
RUN_COOLDOWN_SECONDS = float(_get_env("RUN_COOLDOWN_SECONDS", "10") or "10")
CACHE_TTL_SECONDS = int(_get_env("CACHE_TTL_SECONDS", "3600") or "3600")

# Input & research loop caps
MAX_RESEARCH_PROMPT_CHARS = int(_get_env("MAX_RESEARCH_PROMPT_CHARS", "500") or "500")
MAX_COACH_MESSAGE_CHARS = int(_get_env("MAX_COACH_MESSAGE_CHARS", "400") or "400")
MAX_SEARCH_CALLS = int(_get_env("MAX_SEARCH_CALLS", "3") or "3")
MAX_EXTRACT_CALLS = int(_get_env("MAX_EXTRACT_CALLS", "2") or "2")
MAX_EXTRACT_URLS = 4
MAX_EXTRACT_CHARS_PER_URL = 3000
AGENT_RECURSION_LIMIT = int(_get_env("AGENT_RECURSION_LIMIT", "8") or "8")

# App branding & contact metadata
APP_DOMAIN = "https://www.jobtruth.top"
LINKEDIN_URL = "https://www.linkedin.com/in/abdelrahmanbasry/"
CONTACT_EMAIL = "a.basry404@gmail.com"

# CORS allowlist for GitHub Pages + local dev servers
DEFAULT_CORS_ORIGINS: list[str] = [
    "https://jobtruth.top",
    "https://www.jobtruth.top",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]


def get_cors_origins() -> list[str]:
    extra = _get_env("CORS_ALLOW_ORIGINS", "")
    origins = list(DEFAULT_CORS_ORIGINS)
    if extra:
        for item in extra.split(","):
            cleaned = item.strip().rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins


# 3 Built-in Business Presets (preserved verbatim from original JobTruth app)
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

PRESETS_BY_ID: dict[str, dict[str, str]] = {p["id"]: p for p in PRESETS}
