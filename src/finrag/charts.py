"""Chart/figure → text: the multi-modal step.

Each extracted figure image is sent to Claude vision, which returns a
structured ChartDescription. The description is what gets indexed, so
chart content becomes retrievable by plain-text hybrid search.
"""

from __future__ import annotations

import anthropic

from .llm import RequestRefusedError, image_block, run_structured
from .models import ChartDescription, ElementType, ParsedElement

SYSTEM = """You convert charts and figures from financial reports into faithful textual
descriptions for a retrieval index.

Read every axis label, unit, legend entry, and data point you can. Report only what is
visible — never invent numbers. If the image is a table rendered as an image, transcribe
its content into the description. If the image is not a data visualization (photo, logo,
decoration), say so in `description` and set chart_type to "other"."""


def describe_chart(png_bytes: bytes) -> ChartDescription:
    content = [
        image_block(png_bytes),
        {"type": "text", "text": "Describe this figure from a financial report."},
    ]
    return run_structured(SYSTEM, content, ChartDescription)


def describe_all(elements: list[ParsedElement], progress=None) -> list[ParsedElement]:
    """Fill in .text for every chart element via the vision model. Mutates and returns.

    A figure the API rejects (bad or unreadable image), or one that holds no data
    (a photo or logo), is left without text, so chunking skips it; auth, quota
    and network errors still propagate.
    """
    for el in elements:
        if el.type is ElementType.CHART and el.image_png and not el.text:
            try:
                description = describe_chart(el.image_png)
            except (anthropic.BadRequestError, ValueError, RequestRefusedError):
                continue
            if not description.is_data:
                continue
            el.text = description.as_text()
            if progress:
                progress(el)
    return elements
