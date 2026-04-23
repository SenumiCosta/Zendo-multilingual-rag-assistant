"""End-to-end RAG pipeline helpers.

Stitches loader → chunker → embedder → retriever → generator into two
high-level functions:

    ingest_pdf(path, index_dir)  -> persisted FAISS index + chunks
    rag_pipeline(question, ...)  -> (answer, sources)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from backend.rag import chunker, embedder, generator, loader, retriever
from backend.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000  # rough cap per request to avoid LLM context overflow


def ingest_pdf(
    pdf_path: str | Path,
    index_dir: str | Path,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = chunker.DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Load a PDF, chunk + embed it, and persist a FAISS index.

    Returns the number of chunks indexed.
    """
    t0 = time.perf_counter()
    text = loader.load_pdf(pdf_path)
    log.info("loaded pdf %s: %d chars", pdf_path, len(text))

    chunks = chunker.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks produced from {pdf_path}")

    vectors = embedder.embed(chunks)
    index = retriever.build_index(vectors)
    retriever.save_index(index, chunks, index_dir)
    log.info(
        "indexed %s: %d chunks in %.2fs",
        pdf_path, len(chunks), time.perf_counter() - t0,
    )
    return len(chunks)


def _cap_context(results: list[RetrievalResult], max_chars: int) -> list[str]:
    """Keep top-ranked chunks until max_chars is reached."""
    out: list[str] = []
    running = 0
    for r in results:
        if running + len(r.text) > max_chars:
            break
        out.append(r.text)
        running += len(r.text)
    # Always include at least one chunk if available.
    if not out and results:
        out.append(results[0].text[:max_chars])
    return out


def rag_pipeline(
    question: str,
    index_dir: str | Path,
    k: int = 3,
    history: list[tuple[str, str]] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[RetrievalResult]]:
    """Answer a question using the persisted FAISS index.

    Returns (answer_text, retrieved_sources).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    t0 = time.perf_counter()
    index, chunks = retriever.load_index(index_dir)
    sources = retriever.search(question, index, chunks, k=k)
    context = _cap_context(sources, max_context_chars)
    log.info(
        "retrieved %d/%d chunks (%d chars ctx)",
        len(context), len(sources), sum(len(c) for c in context),
    )

    answer = generator.generate_answer(question, context, history=history)
    log.info("answered in %.2fs", time.perf_counter() - t0)
    return answer, sources
