"""Tests for backend.rag.reranker."""

from __future__ import annotations

import pytest

from backend.rag import reranker
from backend.rag.retriever import RetrievalResult


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reranker._get_reranker.cache_clear()


def test_empty_candidates_returns_empty() -> None:
    assert reranker.rerank("q", [], top_k=5) == []


def test_rerank_reorders_by_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCE:
        def predict(self, pairs):
            # Reverse the input order via descending scores
            return [len(pairs) - i for i in range(len(pairs))]

    monkeypatch.setattr(reranker, "_get_reranker", lambda _n: FakeCE())

    cands = [
        RetrievalResult("first", 0.1, 0),
        RetrievalResult("second", 0.2, 1),
        RetrievalResult("third", 0.3, 2),
    ]
    out = reranker.rerank("q", cands, top_k=2)
    assert [r.text for r in out] == ["first", "second"]


def test_rerank_falls_back_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_n):
        raise RuntimeError("offline")

    monkeypatch.setattr(reranker, "_get_reranker", boom)
    cands = [RetrievalResult("a", 0.5, 0), RetrievalResult("b", 0.4, 1)]
    out = reranker.rerank("q", cands, top_k=1)
    assert len(out) == 1
    assert out[0].text == "a"
