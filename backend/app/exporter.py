from __future__ import annotations

import csv
import io
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.guardrails import is_safe_http_url, sanitize_excel_cell
from app.schemas import REPORT_COLUMNS, ResearchItem, ResearchReport, report_to_rows


def validate_export_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Enforce strict Pydantic ResearchItem validation before serializing .xlsx/.csv exports
    so ungrounded or malformed rows never leak into downloadable files.
    """
    validated: list[dict[str, str]] = []
    for row in rows:
        try:
            item = ResearchItem.model_validate(
                {
                    "name": str(row.get("Name") or "").strip(),
                    "category": str(row.get("Category") or "General").strip(),
                    "key_details": str(row.get("Key Details") or "").strip(),
                    "pricing_or_metrics": str(row.get("Pricing / Metrics") or "Not publicly listed").strip(),
                    "notes": str(row.get("Notes") or "Verified source").strip(),
                    "source_url": str(row.get("Source URL") or "").strip(),
                }
            )
        except Exception:
            continue
        validated.append(
            {
                "Name": item.name.strip(),
                "Category": item.category.strip(),
                "Key Details": item.key_details.strip(),
                "Pricing / Metrics": item.pricing_or_metrics.strip(),
                "Notes": item.notes.strip(),
                "Source URL": str(item.source_url),
            }
        )
    return validated


def export_report_to_excel_bytes(report: ResearchReport) -> bytes:
    """Serialize a post-validated Pydantic ResearchReport directly to .xlsx bytes."""
    return export_rows_to_excel_bytes(report.title, report.executive_summary, report_to_rows(report))


def export_report_to_csv_bytes(report: ResearchReport) -> bytes:
    """Serialize a post-validated Pydantic ResearchReport directly to .csv bytes."""
    return export_rows_to_csv_bytes(report_to_rows(report))


def make_safe_filename(title: str, ext: str = "xlsx") -> str:
    """Create a clean .xlsx or .csv filename from a report title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (title or "").lower()).strip("_")
    slug = slug[:50] or "research_report"
    clean_ext = ext.lstrip(".").lower() or "xlsx"
    return f"{slug}.{clean_ext}"


def export_rows_to_excel_bytes(
    title: str,
    executive_summary: str,
    rows: list[dict[str, str]],
) -> bytes:
    """Generate a formatted, formula-safe .xlsx workbook in memory (zero disk writes)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Research"

    columns = list(REPORT_COLUMNS)
    header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    link_font = Font(color="2563EB", underline="single", size=10)
    body_font = Font(size=10)

    # Header row
    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 26

    # Data rows
    url_col_idx = columns.index("Source URL") + 1
    for row_idx, row_dict in enumerate(rows, start=2):
        for col_idx, col_name in enumerate(columns, start=1):
            raw_val = str(row_dict.get(col_name, "") or "").strip()
            safe_val = sanitize_excel_cell(raw_val)
            cell = ws.cell(row=row_idx, column=col_idx, value=safe_val)
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if col_idx == url_col_idx and isinstance(safe_val, str) and is_safe_http_url(safe_val):
                cell.hyperlink = safe_val
                cell.font = link_font
            else:
                cell.font = body_font

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    col_widths = {
        "Name": 24,
        "Category": 20,
        "Key Details": 44,
        "Pricing / Metrics": 26,
        "Notes": 38,
        "Source URL": 40,
    }
    for col_idx, col_name in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 24)

    # Concise Summary tab (also formula-safe)
    ws_sum = wb.create_sheet(title="Summary")
    summary_rows = [
        ("Field", "Value"),
        ("Report Title", sanitize_excel_cell(title)),
        ("Executive Summary", sanitize_excel_cell(executive_summary)),
        ("Total Verified Rows", len(rows)),
    ]
    for r_idx, (k, v) in enumerate(summary_rows, start=1):
        c1 = ws_sum.cell(row=r_idx, column=1, value=k)
        c2 = ws_sum.cell(row=r_idx, column=2, value=v)
        for c in (c1, c2):
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        if r_idx == 1:
            c1.fill = header_fill
            c2.fill = header_fill
            c1.font = header_font
            c2.font = header_font
        else:
            c1.font = Font(bold=True, size=10)
            c2.font = body_font

    ws_sum.column_dimensions["A"].width = 22
    ws_sum.column_dimensions["B"].width = 80

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def export_rows_to_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    """Generate a UTF-8 BOM, formula-safe CSV in memory (zero disk writes)."""
    text_buf = io.StringIO()
    writer = csv.DictWriter(text_buf, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe_row = {
            col: sanitize_excel_cell(str(row.get(col, "") or "").strip())
            for col in REPORT_COLUMNS
        }
        writer.writerow(safe_row)
    return text_buf.getvalue().encode("utf-8-sig")
