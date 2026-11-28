from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.guardrails import is_safe_http_url

REPORT_COLUMNS: list[str] = [
    "Name",
    "Category",
    "Key Details",
    "Pricing / Metrics",
    "Notes",
    "Source URL",
]


class ResearchItem(BaseModel):
    """A single structured row in a business research spreadsheet."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=160,
        description="Company, competitor, vendor, or entity name.",
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Industry, product tier, segment, or role.",
    )
    key_details: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Qualitative insights, core offering, positioning, or key growth drivers ONLY. "
            "Do NOT repeat exact dollar figures, percentages, or pricing numbers placed in pricing_or_metrics."
        ),
    )
    pricing_or_metrics: str = Field(
        ...,
        min_length=1,
        max_length=250,
        description=(
            "Concise quantitative numbers, pricing tiers, limits, or percentages ONLY "
            "(e.g., 'Free: 300 mins; Pro: $16.99/mo' or 'FY26 Rev: $4.2B (+18% YoY); Margin: 24%'). "
            "No narrative prose. Use 'Not publicly listed' if unavailable."
        ),
    )
    notes: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Concise actionable takeaway, trade-off, or outreach angle without duplicating "
            "Key Details or Pricing / Metrics."
        ),
    )
    source_url: HttpUrl = Field(
        ...,
        description="Exact verified http:// or https:// URL returned by Tavily search or extract during this run.",
    )

    @field_validator("source_url", mode="before")
    @classmethod
    def _enforce_http_scheme(cls, value: object) -> object:
        raw = str(value).strip() if value is not None else ""
        if not is_safe_http_url(raw):
            raise ValueError("source_url must use an http:// or https:// scheme with a valid host.")
        return raw


class ResearchReport(BaseModel):
    """Structured research deliverable guaranteed to render cleanly as a table and .xlsx/.csv export."""

    title: str = Field(
        ...,
        min_length=3,
        max_length=140,
        description="Short, clear title for the research report.",
    )
    executive_summary: str = Field(
        ...,
        min_length=10,
        max_length=800,
        description="2-4 sentence executive summary highlighting the main findings.",
    )
    items: list[ResearchItem] = Field(
        default_factory=list,
        min_length=0,
        max_length=25,
        description=(
            "List of strictly verified, deduplicated research rows. Never pad to hit a quota; "
            "return an empty list [] if zero sources strictly satisfy the user's criteria."
        ),
    )


class FlaggedRow(BaseModel):
    row_index: int
    reasons: list[str]


def normalize_url(url: str) -> str:
    """Normalize a URL for reliable comparison against Tavily-returned URLs."""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _has_materially_different_pricing(pricing_a: str, pricing_b: str) -> bool:
    """Return True when two rows for the same canonical entity have genuinely distinct concrete pricing tiers."""
    a = " ".join((pricing_a or "").strip().lower().split())
    b = " ".join((pricing_b or "").strip().lower().split())
    sparse = {"", "not publicly listed", "n/a", "unknown", "tbd", "none"}
    if a in sparse or b in sparse:
        return False
    return a != b


def verify_and_flag_report(
    report: ResearchReport,
    verified_url_map: dict[str, str],
    exclusions: list[str] | None = None,
    url_content_map: dict[str, str] | None = None,
) -> tuple[ResearchReport, list[FlaggedRow], int]:
    """
    Post-extraction grounding, domain-fallback, negative-exclusion, and SKU deduplication validator:
      1. Drops any row matching negative exclusions.
      2. Drops any row whose entity name (or normalized alias) does not appear in the snippet/content
         of its cited source_url (when url_content_map is supplied).
      3. Allows domain-fallback URL matching ONLY when the canonical entity name is explicitly confirmed
         in retrieved text from that domain; otherwise drops the row.
      4. Deduplicates overlapping rows sharing the same canonical entity/SKU unless pricing differs materially.
      5. Allows clean 0-row abstention and 1-row outputs without padding.
    Returns (cleaned_report, flagged_rows, unverified_dropped_count).
    """
    from app.guardrails import (
        canonical_entity_key,
        entity_grounded_in_text,
        matches_excluded_entity,
    )

    if not report.items:
        return report, [], 0

    domain_fallback: dict[str, str] = {}
    domain_text_map: dict[str, list[str]] = {}
    for norm_u, orig_url in verified_url_map.items():
        if is_safe_http_url(orig_url):
            host = urlparse(orig_url).netloc.lower().removeprefix("www.")
            domain_fallback.setdefault(host, orig_url)
            if url_content_map and norm_u in url_content_map:
                domain_text_map.setdefault(host, []).append(url_content_map[norm_u])

    verified_items: list[ResearchItem] = []
    item_flags: list[list[str]] = []
    seen_entities: dict[str, list[str]] = {}  # canonical_key -> list of kept pricing_or_metrics
    unverified_dropped = 0
    excl_list = exclusions or []

    for item in report.items:
        if excl_list and matches_excluded_entity(item.name, excl_list):
            unverified_dropped += 1
            continue

        raw_url = str(item.source_url)
        if not is_safe_http_url(raw_url):
            unverified_dropped += 1
            continue

        reasons: list[str] = []
        chosen_item: ResearchItem | None = None

        if not verified_url_map:
            chosen_item = item
        else:
            norm = normalize_url(raw_url)
            if norm in verified_url_map and is_safe_http_url(verified_url_map[norm]):
                if url_content_map is not None:
                    cited_text = url_content_map.get(norm, "")
                    if not entity_grounded_in_text(item.name, cited_text):
                        unverified_dropped += 1
                        continue
                chosen_item = item.model_copy(update={"source_url": verified_url_map[norm]})
            else:
                host = urlparse(raw_url).netloc.lower().removeprefix("www.")
                if host in domain_fallback:
                    # Only allow domain-fallback when entity name is explicitly confirmed in retrieved text from that domain
                    if url_content_map is not None:
                        dom_corpus = "\n".join(domain_text_map.get(host, []))
                        if not entity_grounded_in_text(item.name, dom_corpus):
                            unverified_dropped += 1
                            continue
                    chosen_item = item.model_copy(update={"source_url": domain_fallback[host]})
                    reasons.append("Source URL matched by domain fallback — verify exact page")
                else:
                    unverified_dropped += 1
                    continue

        if chosen_item is not None:
            canon_key = canonical_entity_key(chosen_item.name)
            existing_pricings = seen_entities.get(canon_key, [])
            if existing_pricings:
                # Consolidate overlapping rows unless pricing/tier structure differs materially from every kept row
                if not all(
                    _has_materially_different_pricing(chosen_item.pricing_or_metrics, prev_p)
                    for prev_p in existing_pricings
                ):
                    continue
            seen_entities.setdefault(canon_key, []).append(chosen_item.pricing_or_metrics)

            pm_lower = chosen_item.pricing_or_metrics.strip().lower()
            if pm_lower in ("not publicly listed", "n/a", "unknown", "tbd", "none"):
                reasons.append("Pricing or metrics not publicly listed")
            if len(chosen_item.key_details.strip()) < 15:
                reasons.append("Brief key details — may need manual review")
            verified_items.append(chosen_item)
            item_flags.append(reasons)

    if not verified_items:
        abstention_summary = (
            "No verified records strictly matched all criteria, grounding, and URL provenance checks for this request."
        )
        return (
            report.model_copy(
                update={
                    "items": [],
                    "executive_summary": report.executive_summary or abstention_summary,
                }
            ),
            [],
            unverified_dropped,
        )

    flagged_rows: list[FlaggedRow] = [
        FlaggedRow(row_index=idx, reasons=reasons)
        for idx, reasons in enumerate(item_flags)
        if reasons
    ]

    return report.model_copy(update={"items": verified_items}), flagged_rows, unverified_dropped


def report_to_rows(report: ResearchReport) -> list[dict[str, str]]:
    """Convert a validated ResearchReport into clean JSON-serializable row dicts."""
    return [
        {
            "Name": item.name.strip(),
            "Category": item.category.strip(),
            "Key Details": item.key_details.strip(),
            "Pricing / Metrics": item.pricing_or_metrics.strip(),
            "Notes": item.notes.strip(),
            "Source URL": str(item.source_url),
        }
        for item in report.items
    ]


# ---------------------------------------------------------------------------
# FastAPI Request / Response Models
# ---------------------------------------------------------------------------


class PresetOut(BaseModel):
    id: str
    label: str
    badge: str
    description: str
    prompt: str


class PresetsResponse(BaseModel):
    presets: list[PresetOut]
    runs_remaining: int
    replies_remaining: int | None = None
    ga_measurement_id: str | None = None


class ResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    preset_id: str | None = None
    session_id: str | None = None


class ResearchResponse(BaseModel):
    title: str
    executive_summary: str
    columns: list[str]
    rows: list[dict[str, str]]
    flagged_rows: list[FlaggedRow]
    model_used: str
    verified_url_count: int
    unverified_dropped: int
    duration_sec: float
    cached: bool
    filename: str
    runs_remaining: int


class ExportRequest(BaseModel):
    title: str = Field(default="Research Report", min_length=1, max_length=200)
    executive_summary: str = Field(default="", max_length=2000)
    rows: list[dict[str, str]] = Field(..., min_length=1, max_length=100)
    format: Literal["xlsx", "csv"] = "xlsx"


class CoachMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class CoachRequest(BaseModel):
    messages: list[CoachMessage] = Field(..., min_length=1, max_length=20)
    session_id: str | None = None


class CoachResponse(BaseModel):
    reply: str
    suggested_prompt: str | None = None
    model_used: str
    replies_remaining: int | None = None


class DeepResearchStartRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    target_rows: int | None = Field(default=100, ge=0, le=5000)
    session_id: str | None = None


class DeepResearchStartResponse(BaseModel):
    job_id: str
    status: str
    active_providers: list[str]
    target_rows: int | None = None


class DeepResearchStatusResponse(BaseModel):
    job_id: str
    status: str
    wave: int
    queries_completed: int
    urls_discovered: int
    rows_collected: int
    target_rows: int | None = None
    providers_used: list[str]
    providers_exhausted: list[str]
    preview_rows: list[dict[str, str]]
    has_export: bool
    elapsed_seconds: float
    error: str | None = None
    title: str | None = None
    executive_summary: str | None = None
    filename: str | None = None
