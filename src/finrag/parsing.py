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


def extract_tables(path: Path) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                markdown = _table_to_markdown(table)
                if markdown:
                    elements.append(
                        ParsedElement(
                            type=ElementType.TABLE,
                            page=page_no,
                            source=path.name,
                            text=markdown,
                        )
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
