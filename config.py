import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from environment variables first, then Streamlit secrets if available."""
    val = os.getenv(name)
    if val:
        return val.strip()
    try:
        import streamlit as st

        if name in st.secrets:
            secret_val = st.secrets[name]
            if isinstance(secret_val, str) and secret_val.strip():
                return secret_val.strip()
    except Exception:
        pass
    return default


def get_gemini_api_key() -> str | None:
    """Return GEMINI_API_KEY (falling back to GOOGLE_API_KEY for local compatibility)."""
    return _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")


def get_tavily_api_key() -> str | None:
    """Return TAVILY_API_KEY."""
    return _get_secret("TAVILY_API_KEY")


def get_groq_api_key() -> str | None:
    """Return GROQ_API_KEY for cross-provider fallback."""
    return _get_secret("GROQ_API_KEY")


def get_openrouter_api_key() -> str | None:
    """Return OPENROUTER_API_KEY (isolated free key for Prompt Coach assistant)."""
    return _get_secret("OPENROUTER_API_KEY")


def get_ga_measurement_id() -> str | None:
    """Return optional Google Analytics 4 Measurement ID (e.g., G-XXXXXXXXXX)."""
    return _get_secret("GA_MEASUREMENT_ID")


def get_ga_api_secret() -> str | None:
    """Return optional Google Analytics 4 Measurement Protocol API secret."""
    return _get_secret("GA_API_SECRET")


# Cloud-only cross-provider fallback chain:
#   gemini-4-hydrogen (1 fast attempt) -> gemini-3.5-flash-lite (1 fast attempt)
#   -> Groq qwen/qwen3.8-27b -> Groq openai/gpt-oss-120b
PRIMARY_MODEL = _get_secret("GEMINI_PRIMARY_MODEL", "gemini-4-hydrogen")
FALLBACK_MODEL = _get_secret("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
GROQ_FALLBACK_MODEL = _get_secret("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_SECONDARY_FALLBACK_MODEL = _get_secret(
    "GROQ_SECONDARY_FALLBACK_MODEL", "openai/gpt-oss-120b"
)


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

# Fast failover settings (22s Gemini timeout; 40s Groq timeout; 1 fast attempt per model)
LLM_REQUEST_TIMEOUT_SECONDS = float(_get_secret("LLM_REQUEST_TIMEOUT_SECONDS", "22") or "22")
GROQ_REQUEST_TIMEOUT_SECONDS = float(_get_secret("GROQ_REQUEST_TIMEOUT_SECONDS", "40") or "40")
LLM_MAX_RETRIES = 1  # Disables hidden multi-minute google-genai SDK backoff (1 attempt per request)
GEMINI_RETRY_ATTEMPTS = max(1, int(_get_secret("GEMINI_RETRY_ATTEMPTS", "1") or "1"))
GROQ_RETRY_ATTEMPTS = max(1, int(_get_secret("GROQ_RETRY_ATTEMPTS", "1") or "1"))
MODEL_RETRY_MIN_DELAY_SECONDS = float(_get_secret("MODEL_RETRY_MIN_DELAY_SECONDS", "1.5") or "1.5")
MODEL_RETRY_MAX_DELAY_SECONDS = float(_get_secret("MODEL_RETRY_MAX_DELAY_SECONDS", "2.5") or "2.5")
TAVILY_TIMEOUT_SECONDS = float(_get_secret("TAVILY_TIMEOUT_SECONDS", "15") or "15")

# Isolated Prompt Coach assistant settings (OpenRouter free tier only)
OPENROUTER_MODEL = _get_secret("OPENROUTER_MODEL", "openrouter/free") or "openrouter/free"
OPENROUTER_TIMEOUT_SECONDS = float(_get_secret("OPENROUTER_TIMEOUT_SECONDS", "20") or "20")
MAX_COACH_MESSAGES_PER_SESSION = int(_get_secret("MAX_COACH_MESSAGES_PER_SESSION", "10") or "10")

# Guardrails & app metadata
MAX_CUSTOM_RUNS_PER_SESSION = int(_get_secret("MAX_CUSTOM_RUNS_PER_SESSION", "15") or "15")
RUN_COOLDOWN_SECONDS = float(_get_secret("RUN_COOLDOWN_SECONDS", "10") or "10")
MAX_RESEARCH_PROMPT_CHARS = int(_get_secret("MAX_RESEARCH_PROMPT_CHARS", "500") or "500")
MAX_COACH_MESSAGE_CHARS = int(_get_secret("MAX_COACH_MESSAGE_CHARS", "400") or "400")
MAX_SEARCH_CALLS = int(_get_secret("MAX_SEARCH_CALLS", "3") or "3")
MAX_EXTRACT_CALLS = int(_get_secret("MAX_EXTRACT_CALLS", "2") or "2")
MAX_EXTRACT_URLS = 4
MAX_EXTRACT_CHARS_PER_URL = 3000
AGENT_RECURSION_LIMIT = int(_get_secret("AGENT_RECURSION_LIMIT", "8") or "8")
APP_DOMAIN = "https://jobtruth.top"
