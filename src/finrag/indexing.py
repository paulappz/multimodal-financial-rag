"""Qdrant-backed hybrid index (dense + sparse) via LlamaIndex.

Qdrant runs in embedded mode (a local directory) — no server or Docker
required. Hybrid search combines dense semantic vectors (fastembed,
fully local) with sparse BM25 vectors, fused by Qdrant at query time.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from . import config


def _client(storage_dir: Path | None = None) -> QdrantClient:
    path = (storage_dir or config.STORAGE_DIR) / "qdrant"
    path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def _embed_model() -> FastEmbedEmbedding:
    return FastEmbedEmbedding(model_name=config.DENSE_MODEL)


def _vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION,
        enable_hybrid=True,
        fastembed_sparse_model=config.SPARSE_MODEL,
    )


def build_index(
    nodes: list[TextNode], storage_dir: Path | None = None, client: QdrantClient | None = None
) -> VectorStoreIndex:
    client = client or _client(storage_dir)
    store = _vector_store(client)
    storage_context = StorageContext.from_defaults(vector_store=store)
    return VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=_embed_model(),
        show_progress=True,
    )


def load_index(storage_dir: Path | None = None, client: QdrantClient | None = None) -> VectorStoreIndex:
    client = client or _client(storage_dir)
    store = _vector_store(client)
    return VectorStoreIndex.from_vector_store(vector_store=store, embed_model=_embed_model())
