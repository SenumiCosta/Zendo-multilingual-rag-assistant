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


def test_capabilities_reports_multimodal_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes.multimodal, "is_available", lambda: False)
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["multimodal"] is False
    assert body["agentic"] is True


def test_search_image_text_503_when_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes.multimodal, "is_available", lambda: False)
    resp = client.post("/search-image-text", data={"query": "hello"})
    assert resp.status_code == 503


def test_image_endpoint_blocks_non_image_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression: /image must not serve index.faiss or chunks.json."""
    fake_index = tmp_path
    (fake_index / "images").mkdir()
    (fake_index / "index.faiss").write_bytes(b"sensitive")
    (fake_index / "chunks.json").write_text("[]")
    monkeypatch.setattr(routes, "INDEX_DIR", fake_index)

    resp = client.get("/image", params={"path": "index.faiss"})
    assert resp.status_code == 400
    resp = client.get("/image", params={"path": "chunks.json"})
    assert resp.status_code == 400


def test_image_endpoint_blocks_path_traversal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_index = tmp_path / "index"
    (fake_index / "images").mkdir(parents=True)
    monkeypatch.setattr(routes, "INDEX_DIR", fake_index)

    # Try to escape with ..
    resp = client.get("/image", params={"path": "../../../etc/passwd"})
    assert resp.status_code == 400


def test_image_endpoint_serves_valid_image(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_index = tmp_path / "index"
    images = fake_index / "images" / "doc"
    images.mkdir(parents=True)
    target = images / "page_001.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
    monkeypatch.setattr(routes, "INDEX_DIR", fake_index)

    resp = client.get("/image", params={"path": "images/doc/page_001.png"})
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG\r\n\x1a\nfakeimage"


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
    body = resp.json()
    assert body["filename"] == "book.pdf"
    assert body["chunks"] == 7
    assert body["images_indexed"] == 0


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

    def fake_pipeline(question, index_dir, **_kw):
        return (
            "42",
            [RetrievalResult(text="ctx", score=0.9, index=0)],
            {"route": "simple", "total_ms": 12},
        )

    monkeypatch.setattr(routes.pipeline, "rag_pipeline", fake_pipeline)

    resp = client.post("/chat", json={"question": "What is it?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42"
    assert body["sources"] == [{"text": "ctx", "score": 0.9, "index": 0}]
    assert body["meta"]["route"] == "simple"
