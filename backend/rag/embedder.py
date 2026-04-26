"""Chunks -> vectors.

Wraps a SentenceTransformer model for producing multilingual embeddings.
The model is loaded lazily on first use so import is cheap, and cached as
a module-level singleton so subsequent calls reuse the same weights.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE-M3 — 1024-dim, 8K context, strong multilingual coverage including
# Sinhala. Loaded via sentence-transformers (supports dense embeddings
# directly; sparse/multi-vector outputs require FlagEmbedding).
DEFAULT_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Return a cached SentenceTransformer.

    Uses the EMBEDDING_MODEL env var when model_name is not given, falling
    back to the multilingual MiniLM default.
    """
    name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    return SentenceTransformer(name)


def embed(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed a list of texts.

    Args:
        texts: Strings to embed (Sinhala + English both supported).
        model_name: Optional override for the embedding model.

    Returns:
        A float32 ndarray of shape (len(texts), embedding_dim). Returns an
        empty (0, 0) array for empty input.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)
