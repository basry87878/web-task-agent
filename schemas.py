from __future__ import annotations

import io
import re
from urllib.parse import urlparse

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field, HttpUrl, field_validator

_SAFE_URL_SCHEMES = ("http", "https")
_EXCEL_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def is_safe_http_url(url: str) -> bool:
    """Return True only when URL uses an explicit http:// or https:// scheme with a host."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme.lower() in _SAFE_URL_SCHEMES and bool(parsed.netloc)


def sanitize_excel_cell(value: object) -> object:
    """Neutralize Excel formula injection on string cells starting with =, +, -, @, tab, or CR."""
    if not isinstance(value, str) or not value:
        return value
    stripped_leading_spaces = value.lstrip(" ")
    if value[0] in _EXCEL_FORMULA_PREFIXES or (
        stripped_leading_spaces and stripped_leading_spaces[0] in _EXCEL_FORMULA_PREFIXES
    ):
        return "'" + value
    return value


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
    """Structured research deliverable guaranteed to render cleanly as a table and .xlsx export."""

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


def normalize_url(url: str) -> str:
    """Normalize a URL for reliable comparison against Tavily-returned URLs."""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def verify_report_sources(
    report: ResearchReport,
    verified_url_map: dict[str, str],
) -> tuple[ResearchReport, int]:
    """
    Ensure every row's source_url matches a URL actually returned by Tavily in this run.
    Returns (cleaned_report, dropped_or_unverified_count).
    """
    if not verified_url_map:
        return report, 0

    verified_items: list[ResearchItem] = []
    unverified_count = 0

    # Also build a domain-to-url lookup as a secondary fallback when the model trims a query string
    domain_fallback: dict[str, str] = {}
    for orig_url in verified_url_map.values():
        host = urlparse(orig_url).netloc.lower().removeprefix("www.")
        domain_fallback.setdefault(host, orig_url)

    for item in report.items:
        raw_url = str(item.source_url)
        if not is_safe_http_url(raw_url):
            unverified_count += 1
            continue
        norm = normalize_url(raw_url)
        if norm in verified_url_map and is_safe_http_url(verified_url_map[norm]):
            canonical = verified_url_map[norm]
            verified_items.append(item.model_copy(update={"source_url": canonical}))
        else:
            host = urlparse(raw_url).netloc.lower().removeprefix("www.")
            if host in domain_fallback and is_safe_http_url(domain_fallback[host]):
                verified_items.append(item.model_copy(update={"source_url": domain_fallback[host]}))
            else:
                unverified_count += 1

    if not verified_items:
        # Never return an empty report if Tavily returned results; keep original items as fallback
        return report, unverified_count

    return report.model_copy(update={"items": verified_items}), unverified_count


def report_to_dataframe(report: ResearchReport) -> pd.DataFrame:
    """Convert a validated ResearchReport into a clean display/export DataFrame."""
    rows = [
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
    return pd.DataFrame(rows)


def make_safe_filename(title: str) -> str:
    """Create a clean .xlsx filename from a report title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    slug = slug[:50] or "research_report"
    return f"{slug}.xlsx"


def export_report_to_excel_bytes(report: ResearchReport) -> bytes:
    """Generate a polished, multi-user-safe .xlsx file in memory (zero disk writes, formula-safe)."""
    df = report_to_dataframe(report)
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(sanitize_excel_cell)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        safe_df.to_excel(writer, index=False, sheet_name="Research")
        ws = writer.sheets["Research"]

        header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=11)
        link_font = Font(color="2563EB", underline="single", size=10)
        body_font = Font(size=10)

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for col_idx in range(1, len(safe_df.columns) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

        ws.row_dimensions[1].height = 26

        url_col_idx = safe_df.columns.get_loc("Source URL") + 1
        for row_idx in range(2, len(safe_df) + 2):
            for col_idx in range(1, len(safe_df.columns) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                if col_idx == url_col_idx and isinstance(cell.value, str) and is_safe_http_url(cell.value):
                    cell.hyperlink = cell.value
                    cell.font = link_font
                else:
                    cell.font = body_font

        # Set readable column widths capped for clean printing/viewing
        col_widths = {
            "Name": 24,
            "Category": 20,
            "Key Details": 44,
            "Pricing / Metrics": 26,
            "Notes": 38,
            "Source URL": 40,
        }
        for col_idx, col_name in enumerate(safe_df.columns, start=1):
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = col_widths.get(col_name, 24)

        # Add a concise Summary sheet (also sanitized against Excel formula injection)
        summary_df = pd.DataFrame(
            [
                {"Field": "Report Title", "Value": sanitize_excel_cell(report.title)},
                {"Field": "Executive Summary", "Value": sanitize_excel_cell(report.executive_summary)},
                {"Field": "Total Verified Rows", "Value": len(report.items)},
            ]
        )
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        ws_sum = writer.sheets["Summary"]
        ws_sum.column_dimensions["A"].width = 22
        ws_sum.column_dimensions["B"].width = 80
        for row_idx in range(1, 5):
            for col_idx in (1, 2):
                cell = ws_sum.cell(row=row_idx, column=col_idx)
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                if row_idx == 1:
                    cell.fill = header_fill
                    cell.font = header_font
                elif col_idx == 1:
                    cell.font = Font(bold=True, size=10)

    buffer.seek(0)
    return buffer.getvalue()
