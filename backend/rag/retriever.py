"""FAISS similarity search.

Builds an in-memory FAISS index of chunk embeddings and persists it to
disk alongside the original chunk texts. Uses IndexFlatIP (inner product),
which is equivalent to cosine similarity because the embedder L2-normalizes
its vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from backend.rag import embedder

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""

    text: str
    score: float
    index: int


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index from a matrix of embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index: faiss.Index, chunks: list[str], path: str | Path) -> None:
    """Persist the FAISS index and its chunk texts to a directory."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILENAME))
    (out_dir / CHUNKS_FILENAME).write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )


def load_index(path: str | Path) -> tuple[faiss.Index, list[str]]:
    """Load a previously saved FAISS index and its chunk texts."""
    in_dir = Path(path)
    index_path = in_dir / INDEX_FILENAME
    chunks_path = in_dir / CHUNKS_FILENAME
    if not index_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(f"No index found in {in_dir}")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, chunks


def search(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    k: int = 3,
) -> list[RetrievalResult]:
    """Return the top-k most similar chunks to the query."""
    if index.ntotal == 0 or not chunks:
        return []
    k = min(k, index.ntotal)
    query_vec = embedder.embed([query])
    scores, indices = index.search(query_vec, k)
    results: list[RetrievalResult] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        results.append(RetrievalResult(text=chunks[idx], score=float(score), index=int(idx)))
    return results
