"""Tests for backend.rag.embedder.

The real model is heavy to download, so we mock SentenceTransformer. A
separate integration test (marked `slow`) exercises the real model and is
skipped by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.rag import embedder


class _FakeModel:
    """Stand-in for SentenceTransformer with a deterministic output shape."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), self.dim), dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    embedder.get_model.cache_clear()


def test_embed_empty_returns_empty_array() -> None:
    out = embedder.embed([])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 0)


def test_embed_returns_float32_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda _name: _FakeModel(dim=8))

    out = embedder.embed(["hello", "ආයුබෝවන්", "world"])
    assert out.shape == (3, 8)
    assert out.dtype == np.float32


def test_get_model_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def factory(_name: str) -> _FakeModel:
        calls["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(embedder, "SentenceTransformer", factory)

    embedder.get_model("some/model")
    embedder.get_model("some/model")
    assert calls["n"] == 1


@pytest.mark.slow
def test_embed_real_model_integration() -> None:
    """Downloads the real multilingual model. Run with `pytest -m slow`."""
    out = embedder.embed(["hello", "ආයුබෝවන්"])
    assert out.shape[0] == 2
    assert out.shape[1] > 0
