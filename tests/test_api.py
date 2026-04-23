"""Tests for backend/api/routes.py using FastAPI's TestClient.

The heavy dependencies (ingest pipeline, LLM) are monkeypatched so these
tests stay fast and hermetic.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_happy_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(routes, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(
        routes.pipeline, "ingest_pdf", lambda *_a, **_k: 7
    )

    resp = client.post(
        "/upload",
        files={"file": ("book.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"filename": "book.pdf", "chunks": 7}


def test_chat_without_index_returns_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_a, **_k):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(routes.pipeline, "rag_pipeline", missing)
    resp = client.post("/chat", json={"question": "hi"})
    assert resp.status_code == 409


def test_chat_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.rag.retriever import RetrievalResult

    def fake_pipeline(question, index_dir, k=3, history=None):
        return "42", [RetrievalResult(text="ctx", score=0.9, index=0)]

    monkeypatch.setattr(routes.pipeline, "rag_pipeline", fake_pipeline)

    resp = client.post("/chat", json={"question": "What is it?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42"
    assert body["sources"] == [{"text": "ctx", "score": 0.9, "index": 0}]
