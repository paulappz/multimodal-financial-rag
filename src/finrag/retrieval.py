"""Hybrid retrieval over the Qdrant index."""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from . import config
from .models import RetrievedChunk


def retrieve(index: VectorStoreIndex, query: str, top_k: int | None = None,
             documents: list[str] | None = None) -> list[RetrievedChunk]:
    """Hybrid search; `documents` limits it to those source files (e.g. one company's reports)."""
    k = top_k or config.TOP_K
    filters = None
    if documents:
        filters = MetadataFilters(filters=[MetadataFilter(key="source", value=list(documents),
                                                          operator=FilterOperator.IN)])
    retriever = index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=k,
        sparse_top_k=k * 2,
        filters=filters,
    )
    results = retriever.retrieve(query)
    return [
        RetrievedChunk(
            text=r.node.get_content(),
            score=r.score or 0.0,
            source=r.node.metadata.get("source", "?"),
            page=int(r.node.metadata.get("page", 0)),
            element_type=r.node.metadata.get("element_type", "text"),
        )
        for r in results
    ]
