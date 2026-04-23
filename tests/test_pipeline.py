"""Tests for backend.rag.pipeline helpers."""

from __future__ import annotations

import pytest

from backend.rag import pipeline
from backend.rag.retriever import RetrievalResult


def _results(*texts: str) -> list[RetrievalResult]:
    return [RetrievalResult(text=t, score=1.0, index=i) for i, t in enumerate(texts)]


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
