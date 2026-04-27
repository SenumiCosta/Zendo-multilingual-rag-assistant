"""End-to-end RAG pipeline (v0.2).

Ingest:    loader → chunker → embedder → hybrid index (FAISS + BM25)
Query:     router → [HyDE] → hybrid search → [reranker] → LLM

Each stage is optional and controlled by the `Route` chosen for a given
query. Simple questions skip HyDE + reranker for speed; complex/relational
questions get the full 2-stage pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import os

from backend.rag import (
    agentic,
    chunker,
    embedder,
    generator,
    hybrid,
    hyde,
    loader,
    reranker,
    router,
)
from backend.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
CANDIDATE_POOL = 20  # first-stage top-N before rerank


def ingest_pdf(
    pdf_path: str | Path,
    index_dir: str | Path,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = chunker.DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Load PDF → chunk → embed → APPEND to hybrid index → persist.

    If an index already exists in `index_dir`, the new chunks are appended
    so multiple PDFs accumulate in one searchable corpus. Returns the
    number of chunks added by this call.
    """
    import numpy as np

    t0 = time.perf_counter()
    text = loader.load_pdf(pdf_path)
    log.info("loaded pdf %s: %d chars", pdf_path, len(text))

    new_chunks = chunker.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not new_chunks:
        raise ValueError(f"No chunks produced from {pdf_path}")

    new_vectors = embedder.embed(new_chunks)

    # If an index already exists, merge into it; otherwise build fresh.
    try:
        existing = hybrid.load_hybrid_index(index_dir)
        merged_chunks = existing.chunks + new_chunks
        # Reconstruct dense vectors by reindexing — FAISS doesn't expose
        # the underlying matrix from IndexFlatIP cheaply, so we re-embed
        # only the new ones and rebuild the index from (existing + new).
        existing_vectors = _faiss_to_matrix(existing.faiss)
        if existing_vectors.shape[1] != new_vectors.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: existing index is {existing_vectors.shape[1]}-dim, "
                f"new vectors are {new_vectors.shape[1]}-dim. "
                "Clear data/vector_db/ and re-index from scratch."
            )
        merged_vectors = np.concatenate([existing_vectors, new_vectors], axis=0)
        idx = hybrid.build_hybrid_index(merged_chunks, merged_vectors)
        log.info("appending to existing index (%d → %d chunks)", len(existing.chunks), len(merged_chunks))
    except FileNotFoundError:
        idx = hybrid.build_hybrid_index(new_chunks, new_vectors)
        log.info("created new index (%d chunks)", len(new_chunks))

    hybrid.save_hybrid_index(idx, index_dir)
    log.info(
        "indexed %s: +%d chunks in %.2fs (dense+BM25)",
        pdf_path, len(new_chunks), time.perf_counter() - t0,
    )
    return len(new_chunks)


def _faiss_to_matrix(index):
    """Reconstruct the full embedding matrix from a FAISS index."""
    import numpy as np

    n = index.ntotal
    if n == 0:
        return np.empty((0, index.d), dtype=np.float32)
    return index.reconstruct_n(0, n).astype(np.float32)


def _cap_context(results: list[RetrievalResult], max_chars: int) -> list[str]:
    out: list[str] = []
    running = 0
    for r in results:
        if running + len(r.text) > max_chars:
            break
        out.append(r.text)
        running += len(r.text)
    if not out and results:
        out.append(results[0].text[:max_chars])
    return out


def rag_pipeline(
    question: str,
    index_dir: str | Path,
    k: int | None = None,
    history: list[tuple[str, str]] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    force_route: str | None = None,
) -> tuple[str, list[RetrievalResult], dict]:
    """Answer a question using hybrid retrieval + optional HyDE + rerank.

    Returns (answer, sources, meta) where meta records the routing decision
    and per-stage timings for debugging / eval.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    r = router.route(question)
    if force_route:
        r = router.Route(force_route, True, True, 5, "forced")
    final_k = k or r.k
    meta: dict = {"route": r.name, "reason": r.reason}

    # Agentic mode: opt-in via env var, only for complex / relational routes.
    agentic_enabled = os.getenv("AGENTIC_MODE") == "1"
    if agentic_enabled and r.name in {"complex", "relational"}:
        try:
            answer, sources, agentic_meta = agentic.agentic_answer(
                question, index_dir, history=history
            )
            meta.update(agentic_meta)
            return answer, sources, meta
        except Exception as exc:  # noqa: BLE001
            log.warning("agentic path failed (%s); falling back to linear", exc)
            meta["agentic_fallback"] = str(exc)

    t0 = time.perf_counter()
    idx = hybrid.load_hybrid_index(index_dir)

    search_query = question
    if r.use_hyde:
        t_h = time.perf_counter()
        search_query = hyde.hypothetical_document(question)
        meta["hyde_ms"] = int((time.perf_counter() - t_h) * 1000)

    t_s = time.perf_counter()
    candidates = hybrid.hybrid_search(
        idx, search_query, k=CANDIDATE_POOL if r.use_rerank else final_k
    )
    meta["retrieval_ms"] = int((time.perf_counter() - t_s) * 1000)

    if r.use_rerank and candidates:
        t_r = time.perf_counter()
        candidates = reranker.rerank(question, candidates, top_k=final_k)
        meta["rerank_ms"] = int((time.perf_counter() - t_r) * 1000)

    context = _cap_context(candidates, max_context_chars)
    log.info(
        "route=%s retrieved=%d ctx_chars=%d",
        r.name, len(candidates), sum(len(c) for c in context),
    )

    t_g = time.perf_counter()
    answer = generator.generate_answer(question, context, history=history)
    meta["generate_ms"] = int((time.perf_counter() - t_g) * 1000)
    meta["total_ms"] = int((time.perf_counter() - t0) * 1000)
    log.info("answered in %dms meta=%s", meta["total_ms"], meta)
    return answer, candidates, meta
