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
        ...,
        min_length=1,
        max_length=25,
        description="List of structured research rows with verified source URLs.",
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


def verify_and_flag_report(
    report: ResearchReport,
    verified_url_map: dict[str, str],
) -> tuple[ResearchReport, list[FlaggedRow], int]:
    """
    Verify every row's source_url against URLs actually returned by Tavily in this run,
    and identify messy/incomplete rows for visual highlighting in the frontend table.
    Returns (cleaned_report, flagged_rows, unverified_dropped_count).
    """
    domain_fallback: dict[str, str] = {}
    for orig_url in verified_url_map.values():
        if is_safe_http_url(orig_url):
            host = urlparse(orig_url).netloc.lower().removeprefix("www.")
            domain_fallback.setdefault(host, orig_url)

    verified_items: list[ResearchItem] = []
    item_flags: list[list[str]] = []
    unverified_dropped = 0

    for item in report.items:
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
                chosen_item = item.model_copy(update={"source_url": verified_url_map[norm]})
            else:
                host = urlparse(raw_url).netloc.lower().removeprefix("www.")
                if host in domain_fallback:
                    chosen_item = item.model_copy(update={"source_url": domain_fallback[host]})
                    reasons.append("Source URL matched by domain fallback — verify exact page")
                else:
                    unverified_dropped += 1
                    continue

        if chosen_item is not None:
            pm_lower = chosen_item.pricing_or_metrics.strip().lower()
            if pm_lower in ("not publicly listed", "n/a", "unknown", "tbd", "none"):
                reasons.append("Pricing or metrics not publicly listed")
            if len(chosen_item.key_details.strip()) < 15:
                reasons.append("Brief key details — may need manual review")
            verified_items.append(chosen_item)
            item_flags.append(reasons)

    if not verified_items:
        # Never return an empty report if model produced valid items; keep and flag them for review
        fallback_flags = [
            FlaggedRow(
                row_index=idx,
                reasons=["Source URL could not be matched to a Tavily result — verify manually"],
            )
            for idx in range(len(report.items))
        ]
        return report, fallback_flags, unverified_dropped

    # Check for duplicate entity names across surviving rows
    seen_names: dict[str, int] = {}
    flagged_rows: list[FlaggedRow] = []
    for idx, (item, reasons) in enumerate(zip(verified_items, item_flags)):
        norm_name = " ".join(item.name.strip().lower().split())
        if norm_name in seen_names:
            reasons.append("Possible duplicate entity name")
        else:
            seen_names[norm_name] = idx
        if reasons:
            flagged_rows.append(FlaggedRow(row_index=idx, reasons=reasons))

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
    replies_remaining: int
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
    replies_remaining: int
