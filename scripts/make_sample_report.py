"""Generate a synthetic financial report PDF with text, a ruled table, and a chart.

Used by the test suite and as demo input:
    python scripts/make_sample_report.py [output.pdf]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NARRATIVE = [
    "Acme Corporation — Q2 2026 Financial Report",
    "Acme Corporation delivered strong results in the second quarter of 2026. "
    "Total revenue reached $4.2 million, an increase of 31 percent year over year, "
    "driven primarily by growth in the subscription software segment.",
    "Operating expenses grew 12 percent to $2.9 million, reflecting continued "
    "investment in engineering headcount. Net income for the quarter was $0.8 million, "
    "compared with $0.3 million in the second quarter of 2025.",
    "Management raised full-year revenue guidance to a range of $16.5 million to "
    "$17.2 million, citing a record sales pipeline entering the third quarter.",
]

TABLE_ROWS = [
    ["Metric", "Q2 2025", "Q1 2026", "Q2 2026"],
    ["Revenue ($M)", "3.2", "3.8", "4.2"],
    ["Operating expenses ($M)", "2.6", "2.8", "2.9"],
    ["Net income ($M)", "0.3", "0.6", "0.8"],
    ["Gross margin (%)", "61", "64", "66"],
]


def _chart_png() -> bytes:
    quarters = ["Q3'25", "Q4'25", "Q1'26", "Q2'26"]
    revenue = [3.4, 3.6, 3.8, 4.2]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(quarters, revenue, marker="o", linewidth=2, color="#2563eb")
    ax.set_title("Quarterly Revenue Trend ($M)")
    ax.set_ylabel("Revenue ($M)")
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_report(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    story = [Paragraph(NARRATIVE[0], styles["Title"]), Spacer(1, 12)]
    story += [item for p in NARRATIVE[1:] for item in (Paragraph(p, styles["BodyText"]), Spacer(1, 8))]
    story.append(PageBreak())

    story.append(Paragraph("Key Financial Metrics", styles["Heading1"]))
    table = Table(TABLE_ROWS, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story += [table, PageBreak()]

    story.append(Paragraph("Revenue Trend", styles["Heading1"]))
    story.append(Image(io.BytesIO(_chart_png()), width=6 * inch, height=3.2 * inch))
    doc.build(story)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_report.pdf")
    make_report(out)
    print(f"Wrote {out}")
