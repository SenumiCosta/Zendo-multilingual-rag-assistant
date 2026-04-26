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
    """Load PDF → chunk → embed → build hybrid index → persist."""
    t0 = time.perf_counter()
    text = loader.load_pdf(pdf_path)
    log.info("loaded pdf %s: %d chars", pdf_path, len(text))

    chunks = chunker.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks produced from {pdf_path}")

    vectors = embedder.embed(chunks)
    idx = hybrid.build_hybrid_index(chunks, vectors)
    hybrid.save_hybrid_index(idx, index_dir)
    log.info(
        "indexed %s: %d chunks in %.2fs (dense+BM25)",
        pdf_path, len(chunks), time.perf_counter() - t0,
    )
    return len(chunks)


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
