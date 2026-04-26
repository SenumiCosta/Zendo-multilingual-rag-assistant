"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused via RRF.

Production RAG systems combine semantic vector search with lexical keyword
search — dense handles paraphrase / multilingual alignment, BM25 handles
rare terms, IDs, and exact phrases. We fuse their rankings with
Reciprocal Rank Fusion (Cormack et al., 2009), which is parameter-light
and score-scale-agnostic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.rag import embedder
from backend.rag.retriever import (
    CHUNKS_FILENAME,
    INDEX_FILENAME,
    RetrievalResult,
)

BM25_FILENAME = "bm25_corpus.json"


@dataclass
class HybridIndex:
    """Bundle of dense + sparse indices over the same chunk set."""

    faiss: faiss.Index
    bm25: BM25Okapi
    chunks: list[str]
    tokenized: list[list[str]]


# Simple Unicode-aware tokenizer: keeps Sinhala (U+0D80–U+0DFF) and word
# characters. Good enough for BM25; not a full segmenter.
_TOKEN_RE = re.compile(r"[\u0D80-\u0DFF]+|\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def build_hybrid_index(chunks: list[str], vectors: np.ndarray) -> HybridIndex:
    if not chunks or vectors.shape[0] == 0:
        raise ValueError("chunks and vectors must be non-empty")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))

    tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return HybridIndex(faiss=index, bm25=bm25, chunks=chunks, tokenized=tokenized)


def save_hybrid_index(idx: HybridIndex, path: str | Path) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx.faiss, str(out / INDEX_FILENAME))
    (out / CHUNKS_FILENAME).write_text(
        json.dumps(idx.chunks, ensure_ascii=False), encoding="utf-8"
    )
    (out / BM25_FILENAME).write_text(
        json.dumps(idx.tokenized, ensure_ascii=False), encoding="utf-8"
    )


def load_hybrid_index(path: str | Path) -> HybridIndex:
    in_dir = Path(path)
    idx_path = in_dir / INDEX_FILENAME
    chunks_path = in_dir / CHUNKS_FILENAME
    bm25_path = in_dir / BM25_FILENAME
    if not (idx_path.is_file() and chunks_path.is_file()):
        raise FileNotFoundError(f"No index found in {in_dir}")
    faiss_idx = faiss.read_index(str(idx_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if bm25_path.is_file():
        tokenized = json.loads(bm25_path.read_text(encoding="utf-8"))
    else:
        tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return HybridIndex(faiss=faiss_idx, bm25=bm25, chunks=chunks, tokenized=tokenized)


def _dense_search(idx: HybridIndex, query: str, top_n: int) -> list[int]:
    q_vec = embedder.embed([query])
    top_n = min(top_n, idx.faiss.ntotal)
    _, indices = idx.faiss.search(q_vec, top_n)
    return [int(i) for i in indices[0] if i >= 0]


def _sparse_search(idx: HybridIndex, query: str, top_n: int) -> list[int]:
    scores = idx.bm25.get_scores(tokenize(query))
    top_n = min(top_n, len(scores))
    return np.argsort(scores)[::-1][:top_n].tolist()


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """RRF score per doc id. k=60 is the canonical default."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_search(
    idx: HybridIndex,
    query: str,
    k: int = 5,
    candidate_pool: int = 20,
) -> list[RetrievalResult]:
    """Fuse dense + BM25 rankings via RRF and return top-k."""
    if idx.faiss.ntotal == 0 or not idx.chunks:
        return []
    dense_ids = _dense_search(idx, query, candidate_pool)
    sparse_ids = _sparse_search(idx, query, candidate_pool)
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        RetrievalResult(text=idx.chunks[doc_id], score=float(score), index=int(doc_id))
        for doc_id, score in top
    ]
