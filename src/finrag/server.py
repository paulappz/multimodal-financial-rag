"""Web API and UI: upload reports, search, and ask cited questions in the browser.

Run with `finrag serve`. One process owns the embedded Qdrant directory, so
every request shares a single client behind a lock. Ingestion is slow
(parsing, plus a vision call per figure), so it runs in a background thread
and the UI polls the job.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import models as qm

from . import config
from .chunking import build_nodes
from .indexing import _client, build_index, load_index
from .llm import RequestRefusedError
from .models import ElementType, RetrievedChunk
from .parsing import parse_pdf
from .retrieval import retrieve

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = int(os.environ.get("FINRAG_MAX_UPLOAD_MB", "50")) * 1024 * 1024
PAGE_DPI = 110
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ ()-]{0,150}\.pdf$", re.IGNORECASE)


def llm_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return "bedrock"
    return "none"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=30)
    documents: list[str] | None = Field(default=None, max_length=50,
                                        description="Only search these reports (file names); all when omitted")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=30)
    documents: list[str] | None = Field(default=None, max_length=50,
                                        description="Only search these reports (file names); all when omitted")


class Store:
    """Uploaded PDFs, the document manifest and the shared Qdrant client."""

    def __init__(self, storage_dir: Path):
        self.dir = storage_dir
        self.uploads = storage_dir / "uploads"
        self.pages = storage_dir / "images" / "pages"
        self.manifest_path = storage_dir / "documents.json"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.pages.mkdir(parents=True, exist_ok=True)
        self.client = _client(storage_dir)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}

    # documents -----------------------------------------------------------
    def documents(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        return json.loads(self.manifest_path.read_text())

    def _save(self, docs: list[dict]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(docs, indent=2))
        tmp.replace(self.manifest_path)

    def document(self, name: str) -> dict | None:
        return next((d for d in self.documents() if d["name"] == name), None)

    def has_index(self) -> bool:
        return self.client.collection_exists(config.COLLECTION)

    def delete(self, name: str) -> None:
        with self.lock:
            if self.has_index():
                self.client.delete(
                    collection_name=config.COLLECTION,
                    points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                        qm.FieldCondition(key="source", match=qm.MatchValue(value=name))])),
                )
            self._save([d for d in self.documents() if d["name"] != name])
        (self.uploads / name).unlink(missing_ok=True)
        for png in self.pages.glob(f"{name}.p*.png"):
            png.unlink()

    # retrieval -----------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None, documents: list[str] | None = None) -> list[RetrievedChunk]:
        with self.lock:
            if not self.has_index():
                return []
            return retrieve(load_index(client=self.client), query, top_k, documents)

    # ingestion -----------------------------------------------------------
    def start_ingest(self, name: str, vision: bool) -> dict:
        job = {"id": uuid.uuid4().hex[:12], "document": name, "status": "running", "stage": "Queued",
               "error": None, "started": time.time()}
        self.jobs[job["id"]] = job
        threading.Thread(target=self._ingest, args=(job, vision), daemon=True).start()
        return job

    def _ingest(self, job: dict, vision: bool) -> None:
        name = job["document"]
        try:
            job["stage"] = "Parsing text, tables and figures"
            elements = parse_pdf(self.uploads / name)
            counts = {t.value: sum(1 for e in elements if e.type is t) for t in ElementType}
            charts = [e for e in elements if e.type is ElementType.CHART]
            described = failed = not_data = 0
            if charts and vision:
                from .charts import describe_chart

                for i, el in enumerate(charts, 1):
                    job["stage"] = f"Describing figure {i} of {len(charts)} with Claude vision"
                    try:
                        description = describe_chart(el.image_png)
                        if description.is_data:
                            el.text = description.as_text()
                            described += 1
                        else:
                            not_data += 1  # photo or logo: left out of the index
                    except (anthropic.BadRequestError, ValueError, RequestRefusedError) as exc:
                        # One unreadable figure is skipped, not fatal; auth, quota and network errors still fail the job.
                        failed += 1
                        job.setdefault("warnings", []).append(f"Figure on page {el.page} skipped: {exc}")
            elements = [e for e in elements if e.text]
            nodes = build_nodes(elements)
            job["stage"] = f"Indexing {len(nodes)} chunks"
            with self.lock:
                build_index(nodes, client=self.client)
                with pymupdf.open(self.uploads / name) as doc:
                    pages = doc.page_count
                docs = [d for d in self.documents() if d["name"] != name]
                docs.append({
                    "name": name, "pages": pages, "chunks": len(nodes),
                    "text_blocks": counts["text"], "tables": counts["table"],
                    "figures": counts["chart"], "figures_described": described, "figures_failed": failed,
                    "figures_not_data": not_data,
                    "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                self._save(docs)
            job.update(status="done", stage="Done")
        except anthropic.APIError as exc:
            job.update(status="failed", stage="Failed", error=_api_error_text(exc))
            (self.uploads / name).unlink(missing_ok=True)
        except Exception as exc:  # report any parser/index failure to the UI
            job.update(status="failed", stage="Failed", error=f"{type(exc).__name__}: {exc}")
            (self.uploads / name).unlink(missing_ok=True)

    # page images ---------------------------------------------------------
    def page_png(self, name: str, page: int) -> Path:
        out = self.pages / f"{name}.p{page}.png"
        if not out.exists():
            with pymupdf.open(self.uploads / name) as doc:
                if not 1 <= page <= doc.page_count:
                    raise HTTPException(404, "No such page")
                out.write_bytes(doc[page - 1].get_pixmap(dpi=PAGE_DPI).tobytes("png"))
        return out


def _api_error_text(exc: anthropic.APIError) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Claude authentication failed. Set ANTHROPIC_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) and restart."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude rate limit reached. Try again shortly."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude API error {exc.status_code}: {exc.message}"
    return str(exc)


def _chunk_json(n: int, c: RetrievedChunk) -> dict:
    return {"n": n, "text": c.text, "score": round(c.score, 4), "source": c.source,
            "page": c.page, "element_type": c.element_type}


def create_app(storage_dir: Path | None = None) -> FastAPI:
    store = Store(storage_dir or config.STORAGE_DIR)
    app = FastAPI(title="finrag", version="0.1.0")
    app.state.store = store

    def known(name: str) -> dict:
        doc = store.document(name)
        if doc is None:
            raise HTTPException(404, "Unknown document")
        return doc

    @app.get("/api/status")
    def status() -> dict:
        return {"backend": llm_backend(), "model": config.LLM_MODEL, "dense_model": config.DENSE_MODEL,
                "sparse_model": config.SPARSE_MODEL, "top_k": config.TOP_K,
                "documents": len(store.documents())}

    @app.get("/api/documents")
    def documents() -> list[dict]:
        return sorted(store.documents(), key=lambda d: d["ingested_at"], reverse=True)

    @app.post("/api/documents", status_code=202)
    async def upload(file: UploadFile = File(...), vision: bool = Form(True)) -> dict:
        name = Path(file.filename or "").name.strip()
        if not _SAFE_NAME.match(name):
            raise HTTPException(400, "Upload a .pdf file with a plain file name (letters, digits, spaces, . _ - ( )).")
        if store.document(name) or any(j["document"] == name and j["status"] == "running"
                                       for j in store.jobs.values()):
            raise HTTPException(409, f"{name} is already indexed. Delete it first to re-ingest.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "That file is not a PDF.")
        (store.uploads / name).write_bytes(data)
        return store.start_ingest(name, vision and llm_backend() != "none")

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        if job_id not in store.jobs:
            raise HTTPException(404, "Unknown job")
        return store.jobs[job_id]

    @app.delete("/api/documents/{name}", status_code=204)
    def delete(name: str) -> Response:
        known(name)
        store.delete(name)
        return Response(status_code=204)

    @app.get("/api/documents/{name}/file")
    def pdf_file(name: str) -> FileResponse:
        known(name)
        return FileResponse(store.uploads / name, media_type="application/pdf")

    @app.get("/api/documents/{name}/pages/{page}.png")
    def page_image(name: str, page: int) -> FileResponse:
        known(name)
        return FileResponse(store.page_png(name, page), media_type="image/png")

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict:
        chunks = store.retrieve(req.query, req.top_k, req.documents)
        return {"query": req.query, "results": [_chunk_json(i, c) for i, c in enumerate(chunks, 1)]}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        from .answer import answer_question

        chunks = store.retrieve(req.question, req.top_k, req.documents)
        if not chunks:
            raise HTTPException(409, "Nothing is indexed yet. Upload a report first.")
        try:
            result, _ = answer_question(req.question, chunks)
        except anthropic.APIError as exc:
            code = 401 if isinstance(exc, anthropic.AuthenticationError) else 502
            raise HTTPException(code, _api_error_text(exc)) from exc
        cited = sorted({i for i in result.cited_blocks if 1 <= i <= len(chunks)})
        return {"question": req.question, "answer": result.answer,
                "insufficient_context": result.insufficient_context, "cited": cited,
                "context": [_chunk_json(i, c) for i, c in enumerate(chunks, 1)]}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
