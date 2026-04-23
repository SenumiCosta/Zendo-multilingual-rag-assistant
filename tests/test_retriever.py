"""Tests for backend.rag.retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.rag import embedder, retriever


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real SentenceTransformer with a deterministic fake."""

    class Fake:
        def encode(self, texts, **_kw):
            # Deterministic: hash-based unit vectors of dim 4.
            rng = np.random.default_rng([hash(t) & 0xFFFFFFFF for t in texts])
            v = rng.standard_normal((len(texts), 4)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            return v

    embedder.get_model.cache_clear()
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda _n: Fake())


def test_build_index_rejects_empty() -> None:
    with pytest.raises(ValueError):
        retriever.build_index(np.empty((0, 4), dtype=np.float32))


def test_build_and_search_roundtrip() -> None:
    chunks = ["apple pie recipe", "quantum physics notes", "gardening tips"]
    vectors = embedder.embed(chunks)
    index = retriever.build_index(vectors)

    results = retriever.search("apple pie recipe", index, chunks, k=2)
    assert len(results) == 2
    assert results[0].text == "apple pie recipe"
    assert results[0].score >= results[1].score


def test_save_and_load_index(tmp_path: Path) -> None:
    chunks = ["hello", "ආයුබෝවන්", "world"]
    vectors = embedder.embed(chunks)
    index = retriever.build_index(vectors)
    retriever.save_index(index, chunks, tmp_path)

    loaded_index, loaded_chunks = retriever.load_index(tmp_path)
    assert loaded_chunks == chunks
    assert loaded_index.ntotal == 3


def test_load_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        retriever.load_index(tmp_path / "nonexistent")


def test_search_empty_index_returns_empty() -> None:
    # Build with one row then create an empty index manually.
    import faiss

    empty = faiss.IndexFlatIP(4)
    assert retriever.search("q", empty, [], k=3) == []
