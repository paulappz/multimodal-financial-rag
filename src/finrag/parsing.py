"""PDF parsing: narrative text, tables, and chart/figure images.

Three complementary extractors, each doing what it is best at:
- Unstructured partitions narrative text into typed elements (falls back to
  pdfplumber's raw text if unstructured is unavailable),
- pdfplumber extracts tables and converts them to Markdown,
- PyMuPDF pulls embedded raster figures and renders vector-drawing-heavy
  pages (charts drawn as vectors) to PNG for the vision pass.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber
import pymupdf

from .models import ElementType, ParsedElement

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 5_000  # ignore logos/icons


def _normalize(text: str) -> str:
    """Collapse PDF line-wrap artifacts inside a block to plain spaces."""
    return " ".join(text.split())
MIN_DRAWINGS_FOR_CHART_PAGE = 25
MIN_TEXT_BLOCK_CHARS = 40
RENDER_DPI = 150


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    def clean(cell: str | None) -> str:
        return " ".join((cell or "").split())

    cleaned = [[clean(c) for c in row] for row in rows if any(clean(c) for c in row)]
    if len(cleaned) < 2:
        return ""
    header, *body = cleaned
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * width,
    ]
    lines += ["| " + " | ".join((r + [""] * width)[:width]) + " |" for r in body]
    return "\n".join(lines)


# Financial statements are mostly "label  1,234  (56)  —  12%" rows under a period header.
_VALUE = re.compile(r"^(\(?[-–]?[$€£]?\d[\d,]*(\.\d+)?[%x]?\)?|[—–-]|n/?a|nm)$", re.IGNORECASE)
_PERIOD = re.compile(
    r"^(Q[1-4][-–]?(\d{2}|\d{4})|[1-4]Q[-–]?(\d{2}|\d{4})|FY[-–]?\d{2,4}|H[12][-–]?\d{2,4}|(19|20)\d{2}"
    r"|\d{1,2}[-–/][A-Za-z]{3}[-–/]\d{2,4}|YoY|QoQ|TTM|LTM)$",
    re.IGNORECASE,
)
MIN_TABLE_ROWS = 3
MIN_GRID_FILL = 0.5  # below this a ruled-table detection has lost most of its cells


def _line_text(chars: list[dict]) -> str:
    """Rebuild a line from its characters, inserting spaces at real word gaps.

    Letter-spaced headings ("S T A T E M E N T  O F") come out as plain words because the gap
    threshold is relative to the line's own typical letter spacing.
    """
    chars = [c for c in chars if c["text"].strip() or c["text"] == " "]
    out, prev, gaps = [], None, []
    for c in chars:
        if prev is not None and c["text"] != " ":
            gaps.append(c["x0"] - prev["x1"])
        prev = c if c["text"] != " " else prev
    typical = sorted(gaps)[len(gaps) // 2] if gaps else 0
    threshold = max(typical * 2.5, 1.5) if typical > 0.8 else 1.5
    prev = None
    for c in chars:
        if c["text"] == " ":
            continue
        if prev is not None and c["x0"] - prev["x1"] > threshold:
            out.append(" ")
        out.append(c["text"])
        prev = c
    return "".join(out).strip()


def _split_row(line: str) -> tuple[str, list[str]]:
    tokens = line.split()
    i = len(tokens)
    while i > 0 and _VALUE.match(tokens[i - 1]):
        i -= 1
    return " ".join(tokens[:i]), tokens[i:]


def _period_header(line: str) -> list[str] | None:
    tokens = line.split()
    periods = []
    for tok in reversed(tokens):
        if not _PERIOD.match(tok):
            break
        periods.insert(0, tok)
    return periods if len(periods) >= 2 else None


def _text_row_tables(page) -> list[str]:
    """Tables rebuilt from text lines: a period header, then label + numeric-value rows."""
    lines = [_line_text(ln["chars"]) for ln in page.extract_text_lines(return_chars=True)]
    tables: list[str] = []
    header: list[str] | None = None
    caption = ""
    rows: list[list[str]] = []

    def flush() -> None:
        nonlocal rows
        while rows and not any(rows[-1][1:]):  # trailing section labels with no values
            rows.pop()
        if header and sum(1 for r in rows if any(r[1:])) >= MIN_TABLE_ROWS:
            tables.append(_table_to_markdown([[caption or "Line item", *header], *rows]))
        rows = []

    prev = ""
    for line in filter(None, lines):
        periods = _period_header(line)
        if periods:
            flush()
            header = periods
            label = line[: line.rfind(periods[0])].strip()
            caption = label if len(label) > 3 else prev
        elif header:
            label, values = _split_row(line)
            label = re.sub(rf"^{page.page_number}\s+(?=[A-Za-z])", "", label)  # footer page number on the same line
            if len(values) > len(header):
                # Footnote markers such as "(1)" or "(2) (3)" sit between the label and the numbers.
                extra = len(values) - len(header)
                label, values = " ".join([label, *values[:extra]]).strip(), values[extra:]
            if values and not label:
                pass  # page numbers and stray figures
            elif values:
                rows.append([label, *[""] * (len(header) - len(values)), *values])
            elif len(line) <= 120 and not line.endswith("."):
                rows.append([label, *[""] * len(header)])  # section heading inside the table
            else:
                flush()  # a prose paragraph ends the table
                header = None
        prev = line
    flush()
    return tables


def _grid_fill(rows: list[list[str | None]]) -> float:
    cells = [c for r in rows for c in r]
    return sum(1 for c in cells if c and c.strip()) / len(cells) if cells else 0.0


def extract_tables(path: Path) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            grids = page.extract_tables()
            good = [t for t in grids if _grid_fill(t) >= MIN_GRID_FILL]
            if grids and len(good) == len(grids):
                tables = [_table_to_markdown(t) for t in grids]
            else:
                # Ruled detection failed (mostly empty cells) or found nothing: rebuild from text rows,
                # falling back to whatever the ruled detector found if the text has no table shape.
                tables = _text_row_tables(page) or [_table_to_markdown(t) for t in good]
            for markdown in tables:
                if markdown:
                    elements.append(
                        ParsedElement(type=ElementType.TABLE, page=page_no, source=path.name, text=markdown)
                    )
    return elements


def extract_text(path: Path) -> list[ParsedElement]:
    try:
        return _extract_text_unstructured(path)
    except Exception as exc:  # missing optional deps, parser errors
        logger.warning("unstructured parse failed (%s); falling back to pdfplumber text", exc)
        return _extract_text_pdfplumber(path)


def _extract_text_unstructured(path: Path) -> list[ParsedElement]:
    from unstructured.partition.pdf import partition_pdf

    keep = {"NarrativeText", "Title", "ListItem", "UncategorizedText"}
    elements: list[ParsedElement] = []
    for el in partition_pdf(filename=str(path), strategy="fast"):
        if el.category not in keep:
            continue
        text = _normalize(el.text or "")
        # Real headings survive; table-cell fragments misread as titles do not.
        min_chars = 12 if el.category == "Title" else MIN_TEXT_BLOCK_CHARS
        if len(text) < min_chars:
            continue
        elements.append(
            ParsedElement(
                type=ElementType.TEXT,
                page=getattr(el.metadata, "page_number", None) or 1,
                source=path.name,
                text=text,
            )
        )
    if not elements:
        raise ValueError("unstructured returned no usable text elements")
    return elements


def _extract_text_pdfplumber(path: Path) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            for block in text.split("\n\n"):
                block = _normalize(block)
                if len(block) >= MIN_TEXT_BLOCK_CHARS:
                    elements.append(
                        ParsedElement(
                            type=ElementType.TEXT,
                            page=page_no,
                            source=path.name,
                            text=block,
                        )
                    )
    return elements


def extract_figures(path: Path) -> list[ParsedElement]:
    """Embedded raster images, plus rendered pages that are mostly vector drawings."""
    elements: list[ParsedElement] = []
    doc = pymupdf.open(path)
    try:
        for page_no, page in enumerate(doc, start=1):
            got_embedded = False
            for xref, *_ in page.get_images(full=True):
                png = pymupdf.Pixmap(doc, xref)
                if png.n - png.alpha >= 4:  # CMYK etc. -> RGB
                    png = pymupdf.Pixmap(pymupdf.csRGB, png)
                data = png.tobytes("png")
                if len(data) >= MIN_IMAGE_BYTES:
                    got_embedded = True
                    elements.append(
                        ParsedElement(
                            type=ElementType.CHART,
                            page=page_no,
                            source=path.name,
                            image_png=data,
                        )
                    )
            if not got_embedded and len(page.get_drawings()) >= MIN_DRAWINGS_FOR_CHART_PAGE:
                pix = page.get_pixmap(dpi=RENDER_DPI)
                elements.append(
                    ParsedElement(
                        type=ElementType.CHART,
                        page=page_no,
                        source=path.name,
                        image_png=pix.tobytes("png"),
                    )
                )
    finally:
        doc.close()
    return elements


def parse_pdf(path: str | Path) -> list[ParsedElement]:
    path = Path(path)
    elements = extract_text(path) + extract_tables(path) + extract_figures(path)
    elements.sort(key=lambda e: (e.page, e.type.value))
    return elements
