"""Web API tests. The end-to-end flow runs offline: real parsing and hybrid index, mocked Claude."""

import time
from unittest.mock import patch

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from finrag.models import RAGAnswer
from finrag.server import create_app


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("web-store"))) as c:
        yield c


def _upload(client, path, name=None, vision="false"):
    with open(path, "rb") as fh:
        return client.post("/api/documents", files={"file": (name or path.name, fh, "application/pdf")},
                           data={"vision": vision})


def _wait(client, job_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("ingest job did not finish")


def test_ui_and_status_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "finrag" in page.text
    assert client.get("/static/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["backend"] == "none" and status["documents"] == 0


def test_rejects_non_pdf_and_unsafe_names(client, tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("hello")
    assert _upload(client, fake).status_code == 400
    assert _upload(client, fake, name="../../evil.pdf").status_code in (400, 409)  # basename only
    assert _upload(client, fake, name="report.exe").status_code == 400


def test_search_and_ask_before_ingest(client):
    assert client.post("/api/search", json={"query": "revenue"}).json()["results"] == []
    assert client.post("/api/ask", json={"question": "revenue?"}).status_code == 409
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_end_to_end_flow(client, sample_pdf):
    resp = _upload(client, sample_pdf)
    assert resp.status_code == 202
    job = _wait(client, resp.json()["id"])
    assert job["status"] == "done", job

    docs = client.get("/api/documents").json()
    assert [d["name"] for d in docs] == [sample_pdf.name]
    assert docs[0]["chunks"] > 0 and docs[0]["tables"] >= 1 and docs[0]["figures_described"] == 0

    assert _upload(client, sample_pdf).status_code == 409  # no duplicate chunks

    hits = client.post("/api/search", json={"query": "gross margin percentage by quarter", "top_k": 4}).json()["results"]
    assert hits and hits[0]["n"] == 1
    assert any(h["element_type"] == "table" for h in hits)

    png = client.get(f"/api/documents/{sample_pdf.name}/pages/{hits[0]['page']}.png")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    assert client.get(f"/api/documents/{sample_pdf.name}/pages/999.png").status_code == 404
    assert client.get("/api/documents/other.pdf/file").status_code == 404

    fake = RAGAnswer(answer="Revenue was $4.2M [1] [9].", cited_blocks=[1, 9], insufficient_context=False)
    with patch("finrag.answer.run_structured", return_value=fake):
        ans = client.post("/api/ask", json={"question": "What was Q2 2026 revenue?"}).json()
    assert ans["cited"] == [1]  # out-of-range citation dropped
    assert ans["context"][0]["source"] == sample_pdf.name

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    auth = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    with patch("finrag.answer.run_structured", side_effect=auth):
        err = client.post("/api/ask", json={"question": "revenue?"})
    assert err.status_code == 401 and "ANTHROPIC_API_KEY" in err.json()["detail"]

    assert client.delete(f"/api/documents/{sample_pdf.name}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.post("/api/search", json={"query": "revenue"}).json()["results"] == []


def test_rejected_figure_is_skipped_not_fatal(tmp_path, sample_pdf, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")  # makes vision available
    request = httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/v1/messages")
    too_big = anthropic.BadRequestError("image exceeds 5 MB maximum",
                                        response=httpx.Response(400, request=request), body=None)
    with TestClient(create_app(tmp_path / "store")) as c, \
            patch("finrag.charts.describe_chart", side_effect=too_big):
        job = _wait(c, _upload(c, sample_pdf, vision="true").json()["id"])
        assert job["status"] == "done", job
        assert job["warnings"] and "skipped" in job["warnings"][0]
        doc = c.get("/api/documents").json()[0]
    assert doc["figures"] >= 1
    assert doc["figures_described"] == 0 and doc["figures_failed"] == doc["figures"]
    assert doc["chunks"] > 0
