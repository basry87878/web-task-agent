from __future__ import annotations

import re
import threading
import time
from urllib.parse import urlparse

from app.config import (
    MAX_RESEARCH_PROMPT_CHARS,
    RATE_LIMIT_TTL_SECONDS,
    get_apify_token,
    get_brightdata_config,
    get_dataforseo_credentials,
    get_exa_api_key,
    get_gemini_api_key,
    get_google_api_key,
    get_groq_api_key,
    get_openrouter_api_key,
    get_scrapfly_api_key,
    get_serpapi_api_key,
    get_serper_api_key,
    get_tavily_api_key,
)

_SAFE_URL_SCHEMES = ("http", "https")
_EXCEL_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+|any\s+|your\s+|the\s+)?"
        r"(?:previous|prior|above|earlier|system|safety)\s+"
        r"(?:instructions?|prompts?|rules?|directives?|guardrails?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(
        r"\b(?:reveal|show|print|dump|output|repeat|leak|expose|display|tell\s+me)\s+"
        r"(?:your\s+|the\s+)?(?:system\s+|hidden\s+|internal\s+|initial\s+|developer\s+)?"
        r"(?:prompt|instructions?|rules?|source\s+code|config(?:uration)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:print|dump|show|reveal|leak|read|cat|export|list|output)\s+"
        r"(?:your\s+|the\s+)?(?:env|environment(?:\s+variables?)?|\.env|secrets?|credentials?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bos\.environ\b", re.IGNORECASE),
    re.compile(r"\bapi[\s_-]*key\b", re.IGNORECASE),
    re.compile(
        r"\b(?:GEMINI|TAVILY|GROQ|OPENROUTER|GOOGLE|SERPER|EXA|SERPAPI|BRIGHTDATA|SCRAPFLY)_API_KEY\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:DATAFORSEO_PASSWORD|DATAFORSEO_LOGIN|BRIGHTDATA_SERP_ZONE)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:enter\s+developer\s+mode|jailbreak|do\s+anything\s+now)\b",
        re.IGNORECASE,
    ),
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"tvly-[0-9A-Za-z_\-]{8,}"),
    re.compile(r"gsk_[0-9A-Za-z_\-]{8,}"),
    re.compile(r"sk-or-(?:v1-)?[0-9A-Za-z_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[0-9A-Za-z_\-\.]{20,}"),
)

_POLITE_BLOCK_MESSAGE = (
    "For security reasons, that request couldn't be processed. "
    "Please enter a standard B2B market research topic or ask how JobTruth works."
)


def is_safe_http_url(url: str) -> bool:
    """Return True only when URL uses an explicit http:// or https:// scheme with a host."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme.lower() in _SAFE_URL_SCHEMES and bool(parsed.netloc)


def sanitize_excel_cell(value: object) -> object:
    """Neutralize Excel/CSV formula injection on cells starting with =, +, -, @, tab, or CR."""
    if not isinstance(value, str) or not value:
        return value
    stripped_leading = value.lstrip(" ")
    if value[0] in _EXCEL_FORMULA_PREFIXES or (
        stripped_leading and stripped_leading[0] in _EXCEL_FORMULA_PREFIXES
    ):
        return "'" + value
    return value


def validate_user_input(
    text: str,
    *,
    max_chars: int = MAX_RESEARCH_PROMPT_CHARS,
    min_chars: int = 1,
) -> tuple[bool, str]:
    """
    Validate user-supplied text before any LLM or Tavily API call.
    Returns (True, cleaned_text) if allowed, or (False, polite_error_message) if rejected.
    """
    if not isinstance(text, str):
        return False, "Please enter a valid text request."
    cleaned = text.strip()
    if not cleaned:
        return False, "Please enter a non-empty research brief or message."
    if len(cleaned) < min_chars:
        return False, f"Please enter a slightly more detailed request (at least {min_chars} characters)."
    if len(cleaned) > max_chars:
        return (
            False,
            f"Your input is too long ({len(cleaned)} characters). Please keep it under {max_chars} characters.",
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return False, _POLITE_BLOCK_MESSAGE
    return True, cleaned


def redact_secrets(text: str) -> str:
    """Mask any accidental API keys or credentials before returning errors or LLM outputs."""
    if not isinstance(text, str) or not text:
        return text or ""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_KEY]", redacted)
    secret_values: list[str | None] = []
    for getter in (
        get_gemini_api_key,
        get_google_api_key,
        get_groq_api_key,
        get_tavily_api_key,
        get_openrouter_api_key,
        get_serper_api_key,
        get_exa_api_key,
        get_serpapi_api_key,
        get_scrapfly_api_key,
        get_apify_token,
    ):
        try:
            secret_values.append(getter())
        except Exception:
            pass
    try:
        dfs = get_dataforseo_credentials()
        if dfs:
            secret_values.extend(dfs)
    except Exception:
        pass
    try:
        bd = get_brightdata_config()
        if bd:
            secret_values.append(bd[0])
    except Exception:
        pass

    for secret_val in secret_values:
        if secret_val and len(secret_val) >= 8 and secret_val in redacted:
            redacted = redacted.replace(secret_val, "[REDACTED_KEY]")
    return redacted


def wrap_sandbox(tag: str, content: str) -> str:
    """Wrap untrusted user or web text in explicit XML-style tags and escape closing tags."""
    safe = (content or "").replace(f"</{tag}>", f"[/{tag}]").replace(f"<{tag}>", f"[{tag}]")
    return f"<{tag}>\n{safe}\n</{tag}>"


def format_exc(exc: BaseException) -> str:
    """Format an exception with class name and repr, redacting any accidental secrets."""
    return redact_secrets(f"{type(exc).__name__}: {exc!r}")


class SessionRateLimiter:
    """Thread-safe in-memory per-client/session counter with TTL eviction and optional cooldown."""

    def __init__(self, max_count: int, ttl_seconds: int = RATE_LIMIT_TTL_SECONDS) -> None:
        self.max_count = max(1, max_count)
        self.ttl_seconds = max(60, ttl_seconds)
        self._lock = threading.Lock()
        # key -> {"count": int, "expires_at": float, "last_ts": float}
        self._store: dict[str, dict[str, float]] = {}

    def _prune_locked(self, now: float) -> None:
        if len(self._store) > 2000:
            expired = [k for k, v in self._store.items() if v["expires_at"] <= now]
            for k in expired:
                self._store.pop(k, None)

    def _get_entry_locked(self, key: str, now: float) -> dict[str, float]:
        entry = self._store.get(key)
        if entry is None or entry["expires_at"] <= now:
            entry = {"count": 0.0, "expires_at": now + self.ttl_seconds, "last_ts": 0.0}
            self._store[key] = entry
        return entry

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            entry = self._get_entry_locked(key, now)
            return max(0, self.max_count - int(entry["count"]))

    def check_cooldown(self, key: str, cooldown_seconds: float) -> int:
        """Return remaining cooldown seconds (0 if ready)."""
        if cooldown_seconds <= 0:
            return 0
        now = time.monotonic()
        with self._lock:
            entry = self._get_entry_locked(key, now)
            last_ts = entry["last_ts"]
            if last_ts > 0 and (now - last_ts) < cooldown_seconds:
                return max(1, int(cooldown_seconds - (now - last_ts) + 0.999))
            return 0

    def can_consume(self, key: str) -> tuple[bool, int]:
        """Check whether key has remaining quota without incrementing."""
        rem = self.remaining(key)
        return (rem > 0, rem)

    def consume(self, key: str, *, touch_cooldown: bool = True) -> int:
        """Increment usage by 1 and return new remaining count."""
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            entry = self._get_entry_locked(key, now)
            entry["count"] = min(float(self.max_count), entry["count"] + 1.0)
            if touch_cooldown:
                entry["last_ts"] = now
            return max(0, self.max_count - int(entry["count"]))

    def touch(self, key: str) -> None:
        """Record last activity timestamp for cooldown without consuming quota (e.g. cached/preset runs)."""
        now = time.monotonic()
        with self._lock:
            entry = self._get_entry_locked(key, now)
            entry["last_ts"] = now
