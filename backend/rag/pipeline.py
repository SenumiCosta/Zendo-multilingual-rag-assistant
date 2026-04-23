"""End-to-end RAG pipeline helpers.

Stitches loader → chunker → embedder → retriever → generator into two
high-level functions:

    ingest_pdf(path, index_dir)  -> persisted FAISS index + chunks
    rag_pipeline(question, ...)  -> (answer, sources)
"""

from __future__ import annotations

from pathlib import Path

from backend.rag import chunker, embedder, generator, loader, retriever
from backend.rag.retriever import RetrievalResult


def ingest_pdf(
    pdf_path: str | Path,
    index_dir: str | Path,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = chunker.DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Load a PDF, chunk + embed it, and persist a FAISS index.

    Returns the number of chunks indexed.
    """
    text = loader.load_pdf(pdf_path)
    chunks = chunker.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks produced from {pdf_path}")
    vectors = embedder.embed(chunks)
    index = retriever.build_index(vectors)
    retriever.save_index(index, chunks, index_dir)
    return len(chunks)


def rag_pipeline(
    question: str,
    index_dir: str | Path,
    k: int = 3,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[RetrievalResult]]:
    """Answer a question using the persisted FAISS index.

    Returns (answer_text, retrieved_sources).
    """
    index, chunks = retriever.load_index(index_dir)
    sources = retriever.search(question, index, chunks, k=k)
    context = [r.text for r in sources]
    answer = generator.generate_answer(question, context, history=history)
    return answer, sources
