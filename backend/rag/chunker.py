"""Text -> chunks.

Splits long text into overlapping chunks suitable for embedding. Uses
LangChain's RecursiveCharacterTextSplitter, which tries paragraph-, then
sentence-, then word-level separators — a good default for mixed
Sinhala/English text.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults tuned for paraphrase-multilingual-MiniLM-L12-v2 (max 128 tokens).
# 500 chars ~ 100-150 tokens, leaving headroom for the model.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The source text (may contain Sinhala + English).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Characters to overlap between consecutive chunks so
            context isn't lost at boundaries.

    Returns:
        A list of chunk strings. Empty input returns an empty list.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "። ", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)
