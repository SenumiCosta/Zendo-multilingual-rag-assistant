"""Cross-encoder reranker — second-stage refinement.

First-stage retrieval (dense + BM25 + RRF) gives ~20 candidates; the
cross-encoder scores each (query, chunk) pair jointly and re-ranks them,
which measurably improves precision@k in production RAG systems.

Uses `sentence-transformers.CrossEncoder` with a multilingual model by
default. Override via RERANKER_MODEL in .env.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from backend.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)

# BGE-reranker-v2-m3 is multilingual, distilled from BGE-M3, fast on CPU.
# Other options: cross-encoder/ms-marco-MiniLM-L-12-v2 (English only).
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_reranker(model_name: str):
    """Lazy-load the CrossEncoder (singleton)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
    model_name: str | None = None,
) -> list[RetrievalResult]:
    """Rescore candidates with a cross-encoder and return the top_k.

    If the model fails to load (e.g. offline, no disk space), logs a
    warning and returns the input unchanged — hybrid retrieval's RRF
    scores are a reasonable fallback.
    """
    if not candidates:
        return []
    name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER)
    try:
        model = _get_reranker(name)
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker load failed (%s); skipping rerank", exc)
        return candidates[:top_k]

    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    reordered = sorted(zip(candidates, scores, strict=True), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievalResult(text=c.text, score=float(s), index=c.index)
        for c, s in reordered[:top_k]
    ]
