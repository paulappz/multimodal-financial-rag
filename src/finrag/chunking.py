"""Element-aware chunking.

Different element types need different strategies:
- narrative text is sentence-split with overlap (per page, so metadata stays exact),
- tables are indexed whole — splitting a table destroys its meaning,
- chart descriptions are indexed whole, tagged back to their page image.
"""

from __future__ import annotations

from collections import defaultdict

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from . import config
from .models import ElementType, ParsedElement


def build_nodes(elements: list[ParsedElement]) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes: list[TextNode] = []

    text_by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    for el in elements:
        if not el.text:
            continue
        meta = {"source": el.source, "page": el.page, "element_type": el.type.value}
        if el.type is ElementType.TEXT:
            text_by_page[(el.source, el.page)].append(el.text)
        elif el.type is ElementType.TABLE:
            nodes.append(TextNode(text=f"Table (page {el.page}):\n{el.text}", metadata=meta))
        elif el.type is ElementType.CHART:
            nodes.append(TextNode(text=f"Figure (page {el.page}):\n{el.text}", metadata=meta))

    for (source, page), blocks in sorted(text_by_page.items()):
        meta = {"source": source, "page": page, "element_type": ElementType.TEXT.value}
        for chunk in splitter.split_text("\n\n".join(blocks)):
            nodes.append(TextNode(text=chunk, metadata=meta))

    return nodes
