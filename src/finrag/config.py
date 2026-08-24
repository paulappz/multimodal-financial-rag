"""Environment-driven configuration."""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("FINRAG_STORAGE_DIR", "./storage"))
QDRANT_PATH = STORAGE_DIR / "qdrant"
IMAGES_DIR = STORAGE_DIR / "images"

COLLECTION = os.environ.get("FINRAG_COLLECTION", "financial_reports")
COLPALI_COLLECTION = os.environ.get("FINRAG_COLPALI_COLLECTION", "report_pages_colpali")

DENSE_MODEL = os.environ.get("FINRAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.environ.get("FINRAG_SPARSE_MODEL", "Qdrant/bm25")
COLPALI_MODEL = os.environ.get("FINRAG_COLPALI_MODEL", "vidore/colpali-v1.3")

LLM_MODEL = os.environ.get("FINRAG_MODEL", "claude-opus-5")
LLM_FALLBACK_MODEL = os.environ.get("FINRAG_FALLBACK_MODEL", "claude-opus-4-8")
LLM_MAX_TOKENS = int(os.environ.get("FINRAG_MAX_TOKENS", "16000"))

CHUNK_SIZE = int(os.environ.get("FINRAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("FINRAG_CHUNK_OVERLAP", "64"))
TOP_K = int(os.environ.get("FINRAG_TOP_K", "6"))
