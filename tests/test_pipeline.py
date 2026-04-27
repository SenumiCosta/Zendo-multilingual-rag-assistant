"""Tests for backend.rag.pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.rag import embedder, hybrid, pipeline
from backend.rag.retriever import RetrievalResult


def _results(*texts: str) -> list[RetrievalResult]:
    return [RetrievalResult(text=t, score=1.0, index=i) for i, t in enumerate(texts)]


@pytest.fixture
def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def encode(self, texts, **_kw):
            out = np.empty((len(texts), 4), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(hash(t) & 0xFFFFFFFF)
                v = rng.standard_normal(4).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-9
                out[i] = v
            return out

    embedder.get_model.cache_clear()
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda _n: Fake())


def test_cap_context_respects_char_budget() -> None:
    results = _results("a" * 300, "b" * 300, "c" * 300)
    out = pipeline._cap_context(results, max_chars=500)
    assert out == ["a" * 300]  # second chunk would exceed 500


def test_cap_context_always_returns_something() -> None:
    # Even if every chunk exceeds the budget, keep the top one (truncated).
    results = _results("x" * 1000)
    out = pipeline._cap_context(results, max_chars=100)
    assert len(out) == 1
    assert len(out[0]) == 100


def test_cap_context_empty_results() -> None:
    assert pipeline._cap_context([], max_chars=100) == []


def test_rag_pipeline_rejects_empty_question(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError):
        pipeline.rag_pipeline("   ", tmp_path)


def test_ingest_pdf_appends_to_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _stub_embedder
) -> None:
    """Regression: a second ingest must NOT wipe the first PDF's chunks."""

    # Stub loader + chunker to avoid needing real PDFs.
    monkeypatch.setattr(
        pipeline.loader, "load_pdf", lambda p: f"text from {Path(p).stem}"
    )
    monkeypatch.setattr(
        pipeline.chunker,
        "chunk_text",
        lambda text, **_kw: [f"{text} chunk {i}" for i in range(2)],
    )

    pipeline.ingest_pdf("doc_a.pdf", tmp_path)
    idx_after_a = hybrid.load_hybrid_index(tmp_path)
    assert len(idx_after_a.chunks) == 2

    pipeline.ingest_pdf("doc_b.pdf", tmp_path)
    idx_after_b = hybrid.load_hybrid_index(tmp_path)
    assert len(idx_after_b.chunks) == 4
    # Both PDFs' chunks must be present
    joined = " ".join(idx_after_b.chunks)
    assert "doc_a" in joined
    assert "doc_b" in joined


def test_faiss_to_matrix_roundtrip() -> None:
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    idx = hybrid.build_hybrid_index(["a", "b"], vectors)
    recovered = pipeline._faiss_to_matrix(idx.faiss)
    assert recovered.shape == vectors.shape
    np.testing.assert_allclose(recovered, vectors, rtol=1e-5)
