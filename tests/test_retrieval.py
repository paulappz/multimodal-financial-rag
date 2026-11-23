"""End-to-end offline retrieval test: parse -> chunk -> hybrid index -> query.

Runs fully locally (fastembed dense + BM25 sparse in embedded Qdrant);
only the vision/answer LLM steps are absent.
"""

import pytest

from finrag.chunking import build_nodes
from finrag.indexing import build_index
from finrag.parsing import parse_pdf
from finrag.retrieval import retrieve


@pytest.fixture(scope="module")
def index(sample_pdf, tmp_path_factory):
    elements = [e for e in parse_pdf(sample_pdf) if e.text]  # skip undescribed charts
    nodes = build_nodes(elements)
    assert nodes
    return build_index(nodes, storage_dir=tmp_path_factory.mktemp("store"))


def test_hybrid_retrieval_returns_ranked_chunks(index):
    chunks = retrieve(index, "What was total revenue in Q2 2026?", top_k=3)
    assert chunks
    assert chunks[0].score >= chunks[-1].score
    assert any("4.2" in c.text for c in chunks)


def test_table_content_is_retrievable(index):
    chunks = retrieve(index, "gross margin percentage by quarter", top_k=4)
    assert any(c.element_type == "table" for c in chunks)


def test_metadata_travels_through(index):
    chunks = retrieve(index, "full-year revenue guidance", top_k=3)
    assert all(c.source.endswith(".pdf") and c.page >= 1 for c in chunks)


def test_documents_filter_limits_sources(index, sample_pdf):
    hits = retrieve(index, "revenue", top_k=5, documents=[sample_pdf.name])
    assert hits and all(c.source == sample_pdf.name for c in hits)
    assert retrieve(index, "revenue", top_k=5, documents=["some-other-report.pdf"]) == []
