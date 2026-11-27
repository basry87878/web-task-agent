from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup
import httpx

from app.config import (
    get_apify_token,
    get_brightdata_config,
    get_dataforseo_credentials,
    get_exa_api_key,
    get_google_api_key,
    get_groq_api_key,
    get_openrouter_api_key,
    get_scrapfly_api_key,
    get_serpapi_api_key,
    get_serper_api_key,
)
from app.guardrails import is_safe_http_url

MIN_ENRICHED_CHARS = 350
MAX_PAGE_TEXT_CHARS = 3000

_MULTI_PART_TLDS = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.nz",
    "co.jp",
    "com.br",
    "com.mx",
    "co.in",
    "com.sg",
    "co.za",
    "com.tr",
    "com.eg",
    "co.il",
}

_SKIP_EXTENSIONS = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".zip",
    ".mp4",
    ".mp3",
    ".docx",
    ".xlsx",
    ".pptx",
)

_HARD_EXHAUST_CODES = (401, 402, 403)
_QUOTA_MARKERS = (
    "quota",
    "credit",
    "exhausted",
    "insufficient",
    "billing",
    "payment",
    "exceeded",
    "limit reached",
    "out of",
    "subscription",
)

PROVIDER_LABELS: dict[str, str] = {
    "serper": "Serper",
    "exa": "Exa",
    "serpapi": "SerpAPI",
    "dataforseo": "DataForSEO",
    "brightdata": "Bright Data",
    "httpx": "Direct Fetch",
    "scrapfly": "Scrapfly",
    "apify": "Apify",
}


class ProviderExhaustedError(RuntimeError):
    """Raised when a provider returns 401/402/403 or a hard 429 quota exhaustion error."""

    def __init__(self, provider: str, status_code: int, detail: str = "") -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} exhausted (HTTP {status_code}): {detail}")


@dataclass
class DiscoveryHit:
    url: str
    title: str
    snippet: str
    text: str
    provider: str
    root_domain: str


def extract_root_domain(url: str) -> str:
    """Return a normalized root domain (e.g. 'otter.ai', 'bbc.co.uk') for deduplication."""
    if not is_safe_http_url(url):
        return ""
    host = (urlparse(url.strip()).hostname or "").lower().removeprefix("www.")
    if not host:
        return ""
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host
    last_two = ".".join(parts[-2:])
    if last_two in _MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


def clean_html_to_text(raw_html: str, max_chars: int = MAX_PAGE_TEXT_CHARS) -> str:
    """Strip non-content chrome (script, style, nav, footer, header) and return clean main text."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def get_provider_status() -> dict[str, object]:
    """Return a safe boolean readiness summary of Deep Research providers (never exposes keys)."""
    flags = {
        "serper": bool(get_serper_api_key()),
        "exa": bool(get_exa_api_key()),
        "serpapi": bool(get_serpapi_api_key()),
        "dataforseo": bool(get_dataforseo_credentials()),
        "brightdata": bool(get_brightdata_config()),
        "scrapfly": bool(get_scrapfly_api_key()),
        "apify": bool(get_apify_token()),
        "gemini": bool(get_google_api_key()),
        "groq": bool(get_groq_api_key()),
        "openrouter": bool(get_openrouter_api_key()),
    }
    active_discovery = [
        name for name in ("serper", "exa", "serpapi", "dataforseo", "brightdata") if flags[name]
    ]
    active_enrichment = ["httpx"] + [
        name for name in ("scrapfly", "apify") if flags[name]
    ]
    active_llms = [name for name in ("gemini", "groq", "openrouter") if flags[name]]
    return {
        **flags,
        "active_discovery": active_discovery,
        "active_enrichment": active_enrichment,
        "active_llms": active_llms,
        "ready": bool(active_discovery and active_llms),
    }


def _check_exhaustion(provider: str, response: httpx.Response) -> None:
    code = response.status_code
    if code in _HARD_EXHAUST_CODES:
        raise ProviderExhaustedError(provider, code, "auth or billing limit")
    if code == 429:
        body_lower = (response.text or "")[:400].lower()
        if any(marker in body_lower for marker in _QUOTA_MARKERS) or not body_lower:
            raise ProviderExhaustedError(provider, code, "quota exhausted")


class DeepRouter:
    """
    Job-scoped multi-provider discovery and content enrichment router.
    Round-robins across healthy discovery providers, tracks exhausted keys per job,
    deduplicates by root domain, and waterfalls content enrichment from free httpx+BS4
    to Scrapfly and Apify only when snippets lack required depth.
    """

    def __init__(self) -> None:
        self._discovery_order: list[str] = []
        if get_serper_api_key():
            self._discovery_order.append("serper")
        if get_exa_api_key():
            self._discovery_order.append("exa")
        if get_serpapi_api_key():
            self._discovery_order.append("serpapi")
        if get_dataforseo_credentials():
            self._discovery_order.append("dataforseo")
        if get_brightdata_config():
            self._discovery_order.append("brightdata")

        self.exhausted: set[str] = set()
        self.used: set[str] = set()
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()
        self._rate_hits: dict[str, int] = {}
        self._httpx_sem = asyncio.Semaphore(6)
        self._scrapfly_sem = asyncio.Semaphore(2)
        self._apify_sem = asyncio.Semaphore(1)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> DeepRouter:
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                )
            },
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("DeepRouter must be used as an async context manager.")
        return self._client

    def healthy_discovery_providers(self) -> list[str]:
        return [p for p in self._discovery_order if p not in self.exhausted]

    def active_provider_labels(self) -> list[str]:
        all_configured = list(self._discovery_order)
        if get_scrapfly_api_key():
            all_configured.append("scrapfly")
        if get_apify_token():
            all_configured.append("apify")
        return [PROVIDER_LABELS.get(p, p) for p in all_configured]

    def used_provider_labels(self) -> list[str]:
        return [PROVIDER_LABELS.get(p, p) for p in sorted(self.used)]

    def exhausted_provider_labels(self) -> list[str]:
        return [PROVIDER_LABELS.get(p, p) for p in sorted(self.exhausted)]

    async def _next_provider_candidates(self) -> list[str]:
        async with self._rr_lock:
            healthy = self.healthy_discovery_providers()
            if not healthy:
                return []
            start = self._rr_index % len(healthy)
            self._rr_index = (self._rr_index + 1) % len(healthy)
            return healthy[start:] + healthy[:start]

    async def search_query(
        self,
        query: str,
        seen_domains: set[str],
    ) -> list[DiscoveryHit]:
        """
        Execute one discovery query using round-robin provider selection with automatic failover.
        Deduplicates hits by root domain against `seen_domains` (updated in place).
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        candidates = await self._next_provider_candidates()
        for provider in candidates:
            if provider in self.exhausted:
                continue
            try:
                raw_hits = await self._dispatch_search(provider, cleaned_query)
                self.used.add(provider)
                self._rate_hits[provider] = 0
                unique_hits: list[DiscoveryHit] = []
                for hit in raw_hits:
                    if not is_safe_http_url(hit.url):
                        continue
                    dom = hit.root_domain or extract_root_domain(hit.url)
                    if not dom or dom in seen_domains:
                        continue
                    seen_domains.add(dom)
                    hit.root_domain = dom
                    unique_hits.append(hit)
                return unique_hits
            except ProviderExhaustedError:
                self.exhausted.add(provider)
                continue
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                if code in _HARD_EXHAUST_CODES:
                    self.exhausted.add(provider)
                elif code == 429:
                    self._rate_hits[provider] = self._rate_hits.get(provider, 0) + 1
                    if self._rate_hits[provider] >= 2:
                        self.exhausted.add(provider)
                continue
            except Exception:
                continue

        return []

    async def _dispatch_search(self, provider: str, query: str) -> list[DiscoveryHit]:
        if provider == "serper":
            return await self._search_serper(query)
        if provider == "exa":
            return await self._search_exa(query)
        if provider == "serpapi":
            return await self._search_serpapi(query)
        if provider == "dataforseo":
            return await self._search_dataforseo(query)
        if provider == "brightdata":
            return await self._search_brightdata(query)
        return []

    async def _search_serper(self, query: str) -> list[DiscoveryHit]:
        api_key = get_serper_api_key()
        if not api_key:
            return []
        resp = await self.client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 20},
            timeout=18.0,
        )
        _check_exhaustion("serper", resp)
        resp.raise_for_status()
        data = resp.json()
        hits: list[DiscoveryHit] = []
        for item in data.get("organic", []) or []:
            url = (item.get("link") or "").strip()
            if not is_safe_http_url(url):
                continue
            snippet = (item.get("snippet") or "").strip()
            hits.append(
                DiscoveryHit(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    snippet=snippet,
                    text=snippet,
                    provider="serper",
                    root_domain=extract_root_domain(url),
                )
            )
        return hits

    async def _search_exa(self, query: str) -> list[DiscoveryHit]:
        api_key = get_exa_api_key()
        if not api_key:
            return []
        resp = await self.client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "type": "auto",
                "numResults": 15,
                "contents": {"text": {"maxCharacters": 2000}},
            },
            timeout=22.0,
        )
        _check_exhaustion("exa", resp)
        resp.raise_for_status()
        data = resp.json()
        hits: list[DiscoveryHit] = []
        for item in data.get("results", []) or []:
            url = (item.get("url") or "").strip()
            if not is_safe_http_url(url):
                continue
            body_text = re.sub(r"\s+", " ", (item.get("text") or "")).strip()[:2000]
            snippet = body_text[:400]
            hits.append(
                DiscoveryHit(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    snippet=snippet,
                    text=body_text,
                    provider="exa",
                    root_domain=extract_root_domain(url),
                )
            )
        return hits

    async def _search_serpapi(self, query: str) -> list[DiscoveryHit]:
        api_key = get_serpapi_api_key()
        if not api_key:
            return []
        resp = await self.client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": 20, "api_key": api_key},
            timeout=22.0,
        )
        _check_exhaustion("serpapi", resp)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            err_msg = str(data["error"]).lower()
            if any(m in err_msg for m in _QUOTA_MARKERS) or "invalid api key" in err_msg:
                raise ProviderExhaustedError("serpapi", 429, err_msg)
        hits: list[DiscoveryHit] = []
        for item in data.get("organic_results", []) or []:
            url = (item.get("link") or "").strip()
            if not is_safe_http_url(url):
                continue
            snippet = (item.get("snippet") or "").strip()
            hits.append(
                DiscoveryHit(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    snippet=snippet,
                    text=snippet,
                    provider="serpapi",
                    root_domain=extract_root_domain(url),
                )
            )
        return hits

    async def _search_dataforseo(self, query: str) -> list[DiscoveryHit]:
        creds = get_dataforseo_credentials()
        if not creds:
            return []
        resp = await self.client.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
            auth=creds,
            json=[
                {
                    "keyword": query,
                    "location_code": 2840,
                    "language_code": "en",
                    "depth": 20,
                }
            ],
            timeout=25.0,
        )
        _check_exhaustion("dataforseo", resp)
        resp.raise_for_status()
        data = resp.json()
        status_code = data.get("status_code")
        if status_code in (40100, 40200, 40201, 40202, 40204):
            raise ProviderExhaustedError("dataforseo", 402, str(data.get("status_message", "")))

        hits: list[DiscoveryHit] = []
        for task in data.get("tasks", []) or []:
            if task.get("status_code") in (40100, 40200, 40201, 40202, 40204):
                raise ProviderExhaustedError("dataforseo", 402, str(task.get("status_message", "")))
            for result in task.get("result", []) or []:
                for item in result.get("items", []) or []:
                    if item.get("type") != "organic":
                        continue
                    url = (item.get("url") or "").strip()
                    if not is_safe_http_url(url):
                        continue
                    snippet = (item.get("description") or "").strip()
                    hits.append(
                        DiscoveryHit(
                            url=url,
                            title=(item.get("title") or "").strip(),
                            snippet=snippet,
                            text=snippet,
                            provider="dataforseo",
                            root_domain=extract_root_domain(url),
                        )
                    )
        return hits

    async def _search_brightdata(self, query: str) -> list[DiscoveryHit]:
        cfg = get_brightdata_config()
        if not cfg:
            return []
        api_key, zone = cfg
        target_url = f"https://www.google.com/search?q={quote_plus(query)}&num=20&brd_json=1"
        resp = await self.client.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"zone": zone, "url": target_url, "format": "raw"},
            timeout=28.0,
        )
        _check_exhaustion("brightdata", resp)
        resp.raise_for_status()

        raw_text = resp.text or ""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else resp.json()
        except Exception:
            return []
        if not isinstance(data, dict):
            return []

        organic = data.get("organic") or data.get("organic_results") or data.get("results") or []
        hits: list[DiscoveryHit] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            url = (item.get("link") or item.get("url") or "").strip()
            if not is_safe_http_url(url):
                continue
            snippet = (item.get("description") or item.get("snippet") or "").strip()
            hits.append(
                DiscoveryHit(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    snippet=snippet,
                    text=snippet,
                    provider="brightdata",
                    root_domain=extract_root_domain(url),
                )
            )
        return hits

    # -----------------------------------------------------------------------
    # 3-Tier Content Enrichment Waterfall
    # -----------------------------------------------------------------------

    async def enrich_hits(
        self,
        hits: list[DiscoveryHit],
        *,
        max_apify_batch: int = 5,
    ) -> list[DiscoveryHit]:
        """
        Enrich only hits whose existing text lacks depth (< 350 chars).
          Tier 1: Free direct httpx + BeautifulSoup (0 credits burned if >= 350 chars)
          Tier 2: Scrapfly JS/ASP scrape (Semaphore(2))
          Tier 3: Apify batch website-content-crawler for remaining empty high-priority URLs
        """
        needs_enrich = [
            h
            for h in hits
            if len((h.text or "").strip()) < MIN_ENRICHED_CHARS
            and not h.url.lower().split("?")[0].endswith(_SKIP_EXTENSIONS)
        ]
        if not needs_enrich:
            return hits

        # Tier 1: Direct free httpx + BeautifulSoup fetch
        await asyncio.gather(*(self._enrich_tier1_httpx(h) for h in needs_enrich))

        # Tier 2: Scrapfly for pages still < 350 chars
        still_thin = [h for h in needs_enrich if len((h.text or "").strip()) < MIN_ENRICHED_CHARS]
        if still_thin and get_scrapfly_api_key() and "scrapfly" not in self.exhausted:
            await asyncio.gather(*(self._enrich_tier2_scrapfly(h) for h in still_thin[:8]))

        # Tier 3: Apify batch crawl for remaining empty high-priority URLs
        still_empty = [
            h for h in still_thin if len((h.text or "").strip()) < MIN_ENRICHED_CHARS
        ][:max_apify_batch]
        if still_empty and get_apify_token() and "apify" not in self.exhausted:
            await self._enrich_tier3_apify(still_empty)

        return hits

    async def _enrich_tier1_httpx(self, hit: DiscoveryHit) -> None:
        async with self._httpx_sem:
            try:
                resp = await self.client.get(hit.url, timeout=10.0)
                if resp.status_code >= 400:
                    return
                ctype = (resp.headers.get("content-type") or "").lower()
                if "html" not in ctype and "text" not in ctype:
                    return
                cleaned = clean_html_to_text(resp.text[:250_000], max_chars=MAX_PAGE_TEXT_CHARS)
                if len(cleaned) >= MIN_ENRICHED_CHARS:
                    hit.text = cleaned
                    self.used.add("httpx")
            except Exception:
                return

    async def _enrich_tier2_scrapfly(self, hit: DiscoveryHit) -> None:
        api_key = get_scrapfly_api_key()
        if not api_key or "scrapfly" in self.exhausted:
            return
        async with self._scrapfly_sem:
            if "scrapfly" in self.exhausted:
                return
            try:
                resp = await self.client.get(
                    "https://api.scrapfly.io/scrape",
                    params={
                        "key": api_key,
                        "url": hit.url,
                        "render_js": "true",
                        "asp": "true",
                    },
                    timeout=35.0,
                )
                _check_exhaustion("scrapfly", resp)
                resp.raise_for_status()
                data = resp.json()
                html_content = ((data.get("result") or {}).get("content")) or ""
                cleaned = clean_html_to_text(html_content[:350_000], max_chars=MAX_PAGE_TEXT_CHARS)
                if len(cleaned) >= MIN_ENRICHED_CHARS:
                    hit.text = cleaned
                    self.used.add("scrapfly")
            except ProviderExhaustedError:
                self.exhausted.add("scrapfly")
            except Exception:
                return

    async def _enrich_tier3_apify(self, hits: list[DiscoveryHit]) -> None:
        token = get_apify_token()
        if not token or not hits or "apify" in self.exhausted:
            return
        async with self._apify_sem:
            if "apify" in self.exhausted:
                return
            try:
                resp = await self.client.post(
                    "https://api.apify.com/v2/acts/apify~website-content-crawler/run-sync-get-dataset-items",
                    params={"token": token},
                    json={
                        "startUrls": [{"url": h.url} for h in hits],
                        "maxCrawlPages": len(hits),
                        "maxCrawlDepth": 0,
                    },
                    timeout=90.0,
                )
                _check_exhaustion("apify", resp)
                resp.raise_for_status()
                items = resp.json()
                if not isinstance(items, list):
                    return
                by_url = {h.url.rstrip("/").lower(): h for h in hits}
                by_domain = {h.root_domain: h for h in hits if h.root_domain}
                matched_any = False
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_url = (item.get("url") or "").strip()
                    raw_text = (item.get("text") or item.get("markdown") or "").strip()
                    cleaned = re.sub(r"\s+", " ", raw_text)[:MAX_PAGE_TEXT_CHARS]
                    if len(cleaned) < MIN_ENRICHED_CHARS:
                        continue
                    target = by_url.get(item_url.rstrip("/").lower()) or by_domain.get(
                        extract_root_domain(item_url)
                    )
                    if target is not None:
                        target.text = cleaned
                        matched_any = True
                if matched_any:
                    self.used.add("apify")
            except ProviderExhaustedError:
                self.exhausted.add("apify")
            except Exception:
                return
