"""Tests for backend.rag.hybrid (dense+BM25 + RRF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.rag import embedder, hybrid


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def encode(self, texts, **_kw):
            vectors = np.empty((len(texts), 4), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(hash(t) & 0xFFFFFFFF)
                v = rng.standard_normal(4).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-9
                vectors[i] = v
            return vectors

    embedder.get_model.cache_clear()
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda _n: Fake())


def test_tokenize_english_and_sinhala() -> None:
    toks = hybrid.tokenize("Hello, ආයුබෝවන් world!")
    assert "hello" in toks
    assert "world" in toks
    # At least one Sinhala-script token present
    assert any(any("\u0D80" <= ch <= "\u0DFF" for ch in t) for t in toks)


def test_build_hybrid_rejects_empty() -> None:
    with pytest.raises(ValueError):
        hybrid.build_hybrid_index([], np.empty((0, 4), dtype=np.float32))


def test_hybrid_search_finds_exact_term() -> None:
    chunks = [
        "apple pie recipe with cinnamon",
        "quantum physics lecture notes",
        "gardening tips for tomatoes",
    ]
    vectors = embedder.embed(chunks)
    idx = hybrid.build_hybrid_index(chunks, vectors)

    results = hybrid.hybrid_search(idx, "apple pie", k=2, candidate_pool=3)
    assert len(results) >= 1
    assert any("apple" in r.text for r in results)


def test_rrf_combines_rankings() -> None:
    fused = hybrid.reciprocal_rank_fusion([[0, 1, 2], [2, 0, 1]])
    # Doc 0 is top-1 once, top-2 once → highest RRF
    assert max(fused, key=fused.get) == 0


def test_save_load_hybrid_roundtrip(tmp_path: Path) -> None:
    chunks = ["hello world", "ආයුබෝවන් ලෝකය", "goodbye"]
    vectors = embedder.embed(chunks)
    idx = hybrid.build_hybrid_index(chunks, vectors)
    hybrid.save_hybrid_index(idx, tmp_path)

    loaded = hybrid.load_hybrid_index(tmp_path)
    assert loaded.chunks == chunks
    assert loaded.faiss.ntotal == 3
    assert len(loaded.tokenized) == 3
