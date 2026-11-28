from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
import ipaddress
import operator
import re
import socket
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

_NON_RESEARCH_BLOCK_MESSAGE = (
    "This looks like a calculation, unit conversion, or general lookup rather than a B2B/market "
    "research brief. JobTruth is built for source-verified company, pricing, and vendor research—"
    "try comparing specific companies, product tiers, or market segments."
)

# Rule 2: Pure arithmetic, unit/currency conversions, and trivial non-research lookups
_PURE_MATH_RE = re.compile(
    r"^(?:(?:what\s+is|what's|calculate|compute|solve|evaluate)\s+)?"
    r"[\s(\[]*[-+]?\d+(?:[.,]\d+)?(?:\s*(?:[+\-*/^x×÷%]|\bof\b|\btimes\b|\bplus\b|\bminus\b|\bdivided\s+by\b)\s*[\s(\[]*[-+]?\d+(?:[.,]\d+)?[\s)\]]*)+[\s)\]=?!.]*$",
    re.IGNORECASE,
)
_UNIT_CONVERSION_RE = re.compile(
    r"^(?:(?:convert|what\s+is|how\s+much\s+is|how\s+many)\s+)?"
    r"\d+(?:[.,]\d+)?\s*"
    r"(?:km|m|cm|mm|mi|miles?|ft|feet|inch(?:es)?|in|yd|yards?|kg|g|mg|lb|lbs|pounds?|oz|ounces?|"
    r"l|ml|gal|gallons?|c|f|k|celsius|fahrenheit|usd|eur|gbp|jpy|cad|aud|egp|sar|aed|inr|chf)\s+"
    r"(?:to|in|into|=)\s+"
    r"(?:km|m|cm|mm|mi|miles?|ft|feet|inch(?:es)?|in|yd|yards?|kg|g|mg|lb|lbs|pounds?|oz|ounces?|"
    r"l|ml|gal|gallons?|c|f|k|celsius|fahrenheit|usd|eur|gbp|jpy|cad|aud|egp|sar|aed|inr|chf)[?!.\s]*$",
    re.IGNORECASE,
)
_TRIVIAL_LOOKUP_RE = re.compile(
    r"^(?:(?:what\s+(?:time|day|date)\s+is\s+it|tell\s+me\s+a\s+joke|"
    r"what(?:'s|\s+is)\s+the\s+weather(?:\s+in\s+[a-z\s]+)?|who\s+won\s+the\s+game)[?!.\s]*)$",
    re.IGNORECASE,
)

_EXCLUSION_CLAUSE_RE = re.compile(
    r"\b(?:exclude|excluding|except|without|not\s+including)\s+([a-zA-Z0-9][a-zA-Z0-9 .&/\-,+]{1,140}?)(?=[.;!?\n]|$|\b(?:and\s+(?:include|provide|compare|focus|list|show|capture|return)|for\s+each|with\s+verified))",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "that", "this", "these", "those",
        "are", "was", "were", "have", "has", "had", "not", "but", "can", "will", "all",
        "any", "each", "their", "them", "they", "our", "your", "about", "what", "which",
        "who", "how", "when", "where", "why", "top", "best", "list", "find", "compare",
        "build", "create", "shortlist", "show", "give", "provide", "include", "including",
        "pricing", "price", "prices", "cost", "costs", "company", "companies", "vendor",
        "vendors", "provider", "providers", "platform", "platforms", "software", "tool",
        "tools", "service", "services", "solution", "solutions", "official", "website",
        "urls", "url", "page", "pages", "verified", "exclude", "excluding", "except",
        "without", "founded", "2024", "2025", "2026", "b2b", "saas",
    }
)

_ENTITY_SUFFIX_RE = re.compile(
    r"\b(?:inc\.?|ltd\.?|llc\.?|gmbh|ag|s\.a\.?|plc|corp\.?|corporation|co\.?|company|"
    r"group|holdings?|technologies|technology|solutions?|services?|systems?|software|"
    r"platform|cloud|enterprise|pro|starter|basic|business|team|plus|plan|tier|edition|"
    r"phase\s*\d+|region(?:al)?|eu|uk|us|emea|apac|na)\b",
    re.IGNORECASE,
)

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class IntentDecision:
    action: str  # "allow" | "short_circuit" | "block_injection" | "invalid"
    category: str  # "research" | "math" | "conversion" | "gibberish" | "non_research" | "injection" | "invalid"
    message: str
    cleaned_prompt: str = ""
    calculation_result: str | None = None


def _safe_eval_arithmetic(text: str) -> str | None:
    """Safely evaluate a pure basic arithmetic expression (e.g., '45646x454?') using AST."""
    expr = re.sub(
        r"^(?:what\s+is|what's|calculate|compute|solve|evaluate)\s+",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    )
    expr = expr.rstrip("?!.= \t")
    expr = re.sub(r"\btimes\b", "*", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bplus\b", "+", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bminus\b", "-", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bdivided\s+by\b", "/", expr, flags=re.IGNORECASE)
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
    expr = expr.replace(",", "")
    if not re.fullmatch(r"[\d.\s+\-*/()]+", expr):
        return None
    try:
        node = ast.parse(expr, mode="eval").body

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_UNARYOPS:
                return _SAFE_UNARYOPS[type(n.op)](_eval(n.operand))
            if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BINOPS:
                left = _eval(n.left)
                right = _eval(n.right)
                if isinstance(n.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("Exponent too large")
                if isinstance(n.op, (ast.Div, ast.Mod)) and right == 0:
                    raise ZeroDivisionError()
                return _SAFE_BINOPS[type(n.op)](left, right)
            raise ValueError("Unsupported AST node")

        val = _eval(node)
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.6g}"
    except Exception:
        return None


def is_gibberish_query(text: str) -> bool:
    """Detect keyboard-mash / non-word gibberish without false-positiving compact valid queries."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return True
    alpha_chars = re.findall(r"[a-zA-Z]", cleaned)
    if len(alpha_chars) < 3:
        return True
    # Excessive repeated single character (e.g., "aaaaaaaaaa")
    if re.search(r"(.)\1{5,}", cleaned.lower()):
        return True
    words = re.findall(r"[a-zA-Z]{2,}", cleaned.lower())
    if not words:
        return True
    vowels = set("aeiouy")
    # Every token has length >= 6 and zero vowels (or > 75% consonant-only long strings)
    long_vowelless = sum(1 for w in words if len(w) >= 5 and not (set(w) & vowels))
    if long_vowelless == len(words):
        return True
    # Keyboard-smash heuristic: single long nonsense token with no spaces and very low vowel ratio
    if len(words) == 1 and len(words[0]) >= 10:
        vowel_ratio = sum(1 for ch in words[0] if ch in vowels) / len(words[0])
        if vowel_ratio < 0.15:
            return True
    return False


def is_non_research_query(text: str) -> bool:
    """Return True if the query is pure arithmetic, unit/currency conversion, or trivial lookup."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    return bool(
        _PURE_MATH_RE.match(cleaned)
        or _UNIT_CONVERSION_RE.match(cleaned)
        or _TRIVIAL_LOOKUP_RE.match(cleaned)
    )


def classify_research_intent(
    text: str,
    *,
    max_chars: int = MAX_RESEARCH_PROMPT_CHARS,
    min_chars: int = 3,
) -> IntentDecision:
    """
    Lightweight pre-search input router run BEFORE Tavily, Serper, Exa, DataForSEO,
    Apify, Bright Data, Scrapfly, or SerpAPI are ever called.
    """
    if not isinstance(text, str):
        return IntentDecision("invalid", "invalid", "Please enter a valid text request.")
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return IntentDecision("invalid", "invalid", "Please enter a non-empty research brief.")
    if len(cleaned) > max_chars:
        return IntentDecision(
            "invalid",
            "invalid",
            f"Your input is too long ({len(cleaned)} characters). Please keep it under {max_chars} characters.",
        )

    # 1. Prompt injection / secret exfiltration -> hard block
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return IntentDecision("block_injection", "injection", _POLITE_BLOCK_MESSAGE)

    # 2. Pure math expression -> short-circuit with safe calculation & 0 external rows
    if _PURE_MATH_RE.match(cleaned):
        calc = _safe_eval_arithmetic(cleaned)
        if calc is not None:
            msg = (
                f"Direct calculation: {cleaned.rstrip('?').strip()} = {calc}. "
                "No web search or external table rows were generated because this is a pure math expression, not a B2B/market research brief."
            )
        else:
            msg = _NON_RESEARCH_BLOCK_MESSAGE
        return IntentDecision("short_circuit", "math", msg, cleaned_prompt=cleaned, calculation_result=calc)

    # 3. Pure unit or currency conversion -> short-circuit with 0 external rows
    if _UNIT_CONVERSION_RE.match(cleaned):
        return IntentDecision(
            "short_circuit",
            "conversion",
            "Out of scope for B2B market research: this query is a unit/currency conversion rather than a company, pricing, or vendor research brief. Zero external sources were queried.",
            cleaned_prompt=cleaned,
        )

    # 4. Trivial non-research lookup -> short-circuit with 0 external rows
    if _TRIVIAL_LOOKUP_RE.match(cleaned):
        return IntentDecision(
            "short_circuit",
            "non_research",
            _NON_RESEARCH_BLOCK_MESSAGE,
            cleaned_prompt=cleaned,
        )

    # 5. Gibberish / keyboard smash -> short-circuit with 0 external rows
    if is_gibberish_query(cleaned):
        return IntentDecision(
            "short_circuit",
            "gibberish",
            "No valid research topic was detected in your input. Please enter a clear B2B market, pricing, or vendor research brief.",
            cleaned_prompt=cleaned,
        )

    if len(cleaned) < min_chars:
        return IntentDecision(
            "invalid",
            "invalid",
            f"Please enter a slightly more detailed request (at least {min_chars} characters).",
        )

    return IntentDecision("allow", "research", "", cleaned_prompt=cleaned)


def extract_negative_exclusions(prompt: str) -> list[str]:
    """Extract explicitly excluded entity names from phrases like 'exclude SAP, Oracle, and Manhattan Associates'."""
    if not prompt or not isinstance(prompt, str):
        return []
    exclusions: list[str] = []
    for match in _EXCLUSION_CLAUSE_RE.finditer(prompt):
        clause = match.group(1)
        parts = re.split(r",|\band\b|\bor\b|/", clause, flags=re.IGNORECASE)
        for part in parts:
            cleaned = re.sub(r"^(?:the|any|all)\s+", "", part.strip(" \"'()—–-"), flags=re.IGNORECASE).strip()
            if 2 <= len(cleaned) <= 60 and cleaned.lower() not in ("others", "more", "etc", "them", "these"):
                exclusions.append(cleaned)
    return list(dict.fromkeys(exclusions))


def matches_excluded_entity(entity_name: str, exclusions: list[str]) -> bool:
    """Return True if an extracted entity name matches any user-specified negative exclusion."""
    if not entity_name or not exclusions:
        return False
    norm_name = " ".join(entity_name.strip().lower().split())
    for excl in exclusions:
        norm_excl = " ".join(excl.strip().lower().split())
        if not norm_excl:
            continue
        if norm_name == norm_excl or re.search(rf"\b{re.escape(norm_excl)}\b", norm_name):
            return True
    return False


def canonical_entity_key(name: str) -> str:
    """Normalize an entity/product name to detect overlapping SKU, legal-suffix, or rollout duplicates."""
    lower = (name or "").strip().lower()
    lower = re.sub(r"\([^)]*\)", " ", lower)  # drop parenthetical regional/tier qualifiers
    lower = _ENTITY_SUFFIX_RE.sub(" ", lower)
    lower = re.sub(r"[^a-z0-9]+", " ", lower).strip()
    return lower or re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def entity_grounded_in_text(entity_name: str, source_text: str) -> bool:
    """
    Return True ONLY when `entity_name` (or its normalized core alias) explicitly appears
    in the retrieved snippet/page content of its cited source.
    """
    if not entity_name or not source_text:
        return False
    haystack = " ".join(source_text.lower().split())
    if not haystack:
        return False

    candidates: list[str] = []
    full_norm = " ".join(entity_name.strip().lower().split())
    if full_norm:
        candidates.append(full_norm)
    stripped = canonical_entity_key(entity_name)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    # Also allow primary brand segment before dash/colon/slash (e.g., "Stripe — Terminal" -> "stripe")
    head = re.split(r"[:\-–—/|]", entity_name.strip().lower())[0].strip()
    head_clean = canonical_entity_key(head)
    if len(head_clean) >= 3 and head_clean not in candidates:
        candidates.append(head_clean)

    for cand in candidates:
        if len(cand) < 2:
            continue
        if re.search(rf"\b{re.escape(cand)}\b", haystack):
            return True
        # Handle domain-style brands like "Otter.ai" vs "otter" or "tl;dv"
        compact = re.sub(r"[^a-z0-9]", "", cand)
        if len(compact) >= 4 and compact in re.sub(r"[^a-z0-9]", "", haystack):
            return True

    # Multi-word entity fallback: require all significant words (>=3 chars) to appear in text
    sig_words = [w for w in re.findall(r"[a-z0-9]{3,}", stripped) if w not in _STOPWORDS]
    if len(sig_words) >= 2 and all(re.search(rf"\b{re.escape(w)}\b", haystack) for w in sig_words):
        return True
    return False


def extract_query_terms(query: str, exclusions: list[str] | None = None) -> set[str]:
    """Extract informative lowercase search terms from a prompt/query, omitting stopwords and excluded names."""
    excl_tokens: set[str] = set()
    for ex in exclusions or []:
        excl_tokens.update(re.findall(r"[a-z0-9]{2,}", ex.lower()))
    tokens = re.findall(r"[a-z0-9]{2,}", (query or "").lower())
    terms = {t for t in tokens if t not in _STOPWORDS and t not in excl_tokens and not t.isdigit()}
    return terms


def score_search_hit_relevance(
    query_terms: set[str],
    *,
    title: str = "",
    snippet: str = "",
    text: str = "",
    url: str = "",
    provider_score: float | None = None,
) -> tuple[float, int]:
    """
    Compute (relevance_score, matched_term_count) for a raw search hit.
    Used to filter noisy off-topic snippets BEFORE entering the LLM extractor window.
    """
    if not query_terms:
        return (provider_score if provider_score is not None else 1.0), 1

    title_l = (title or "").lower()
    body_l = f"{title_l} {(snippet or '').lower()} {(text or '')[:1500].lower()} {(url or '').lower()}"
    matched = 0
    title_hits = 0
    for term in query_terms:
        pat = rf"\b{re.escape(term)}"
        if re.search(pat, body_l):
            matched += 1
            if re.search(pat, title_l):
                title_hits += 1

    overlap_ratio = matched / max(1, len(query_terms))
    title_bonus = 0.15 * (title_hits / max(1, len(query_terms)))
    score = overlap_ratio + title_bonus
    if provider_score is not None:
        score = 0.65 * score + 0.35 * max(0.0, min(1.0, float(provider_score)))
    return round(score, 4), matched


def filter_relevant_search_hits(
    query: str,
    hits: list,
    *,
    exclusions: list[str] | None = None,
    min_overlap: int = 1,
    min_score: float = 0.12,
) -> list:
    """
    Filter raw search hits (dicts or objects with title/snippet/text/url/score) by relevance score
    and query-term overlap before passing chunks to the LLM extractor.
    """
    if not hits:
        return []
    query_terms = extract_query_terms(query, exclusions=exclusions)
    if not query_terms:
        return hits

    scored: list[tuple[float, object]] = []
    required_overlap = min(min_overlap, len(query_terms))
    for hit in hits:
        if isinstance(hit, dict):
            title = str(hit.get("title") or "")
            snippet = str(hit.get("content") or hit.get("snippet") or "")
            text = str(hit.get("text") or snippet)
            url = str(hit.get("url") or "")
            p_score = hit.get("score")
        else:
            title = str(getattr(hit, "title", "") or "")
            snippet = str(getattr(hit, "snippet", "") or "")
            text = str(getattr(hit, "text", "") or snippet)
            url = str(getattr(hit, "url", "") or "")
            p_score = getattr(hit, "score", None)

        rel_score, matched = score_search_hit_relevance(
            query_terms,
            title=title,
            snippet=snippet,
            text=text,
            url=url,
            provider_score=float(p_score) if isinstance(p_score, (int, float)) else None,
        )
        if matched >= required_overlap and rel_score >= min_score:
            scored.append((rel_score, hit))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata.goog",
        "metadata",
        "instance-data",
    }
)
_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".localdomain",
    ".home.arpa",
)


def _is_forbidden_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if an IP is loopback, private, link-local, metadata, multicast, or unspecified."""
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
        or str(ip_obj) == "169.254.169.254"
    )


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse standard, IPv6, decimal-integer, or hex IP literals (catching SSRF obfuscations)."""
    cleaned = host.strip("[]")
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        pass
    # Decimal integer IP (e.g. 2130706433 -> 127.0.0.1) or hex IP (0x7f000001)
    if re.fullmatch(r"\d+", cleaned):
        try:
            num = int(cleaned, 10)
            if 0 <= num <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(num)
        except ValueError:
            pass
    if re.fullmatch(r"0x[0-9a-fA-F]+", cleaned):
        try:
            num = int(cleaned, 16)
            if 0 <= num <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(num)
        except ValueError:
            pass
    return None


def is_safe_http_url(url: str, *, resolve_dns: bool = False) -> bool:
    """
    SSRF-safe URL validator:
      - Requires explicit http:// or https:// scheme and non-empty hostname (no userinfo credentials).
      - Blocks localhost, cloud metadata hostnames, .local/.internal domains.
      - Blocks private (10/8, 172.16/12, 192.168/16), loopback (127/8, ::1), link-local
        (169.254/16, fe80::/10), IPv4-mapped IPv6, decimal/hex IP literals, and reserved ranges.
      - When resolve_dns=True (used right before outbound HTTP fetch), resolves DNS and rejects
        any hostname whose A/AAAA records point to a private/internal IP.
    """
    if not isinstance(url, str):
        return False
    raw = url.strip()
    if not raw or any(ch in raw for ch in ("\r", "\n", "\x00")):
        return False
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False

    if parsed.scheme.lower() not in _SAFE_URL_SCHEMES:
        return False
    if parsed.username or parsed.password:
        return False

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in _BLOCKED_HOSTNAMES or any(host.endswith(sfx) for sfx in _BLOCKED_HOST_SUFFIXES):
        return False

    ip_literal = _parse_ip_literal(host)
    if ip_literal is not None:
        return not _is_forbidden_ip(ip_literal)

    # Reject octal dotted-quad SSRF tricks like 0177.0.0.1
    if re.fullmatch(r"[0-9a-fA-FxX.]+", host) and all(
        re.fullmatch(r"(?:0x[0-9a-fA-F]+|0[0-7]+|\d+)", part) for part in host.split(".") if part
    ):
        return False

    if "." not in host:
        return False

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme.lower() == "https" else 80))
            if not infos:
                return False
            for info in infos:
                addr = info[4][0]
                ip_obj = ipaddress.ip_address(addr.split("%")[0])
                if _is_forbidden_ip(ip_obj):
                    return False
        except OSError:
            return False

    return True


def sanitize_excel_cell(value: object) -> object:
    """Neutralize Excel/CSV formula injection on cells starting with =, +, -, @, tab, CR, or LF."""
    if not isinstance(value, str) or not value:
        return value
    stripped_leading = value.lstrip(" \t\r\n")
    if value[0] in _EXCEL_FORMULA_PREFIXES or value[0] == "\n" or (
        stripped_leading and stripped_leading[0] in _EXCEL_FORMULA_PREFIXES
    ):
        return "'" + value
    return value


def validate_user_input(
    text: str,
    *,
    max_chars: int = MAX_RESEARCH_PROMPT_CHARS,
    min_chars: int = 1,
    reject_non_research: bool = False,
) -> tuple[bool, str]:
    """
    Validate user-supplied text before any LLM or web-search API call.
    Returns (True, cleaned_text) if allowed, or (False, polite_error_message) if rejected.
    """
    if not isinstance(text, str):
        return False, "Please enter a valid text request."
    cleaned = text.strip()
    if not cleaned:
        return False, "Please enter a non-empty research brief or message."
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return False, _POLITE_BLOCK_MESSAGE
    if reject_non_research and is_non_research_query(cleaned):
        return False, _NON_RESEARCH_BLOCK_MESSAGE
    if len(cleaned) < min_chars:
        return False, f"Please enter a slightly more detailed request (at least {min_chars} characters)."
    if len(cleaned) > max_chars:
        return (
            False,
            f"Your input is too long ({len(cleaned)} characters). Please keep it under {max_chars} characters.",
        )
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


class SlidingWindowIPRateLimiter:
    """Thread-safe per-IP sliding-window rate limiter for /api/ endpoints."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1, window_seconds)
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, ip: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if len(self._buckets) > 5000:
                stale = [k for k, q in self._buckets.items() if not q or q[-1] <= cutoff]
                for k in stale:
                    self._buckets.pop(k, None)
            q = self._buckets.get(ip)
            if q is None:
                q = deque()
                self._buckets[ip] = q
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - q[0]) + 0.999))
                return False, retry_after
            q.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
