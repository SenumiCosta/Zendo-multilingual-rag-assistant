"""Tests for backend.rag.chunker."""

from __future__ import annotations

from backend.rag import chunker


def test_chunk_empty_returns_empty() -> None:
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\n  ") == []


def test_chunk_short_text_single_chunk() -> None:
    text = "Hello world."
    chunks = chunker.chunk_text(text, chunk_size=500)
    assert chunks == [text]


def test_chunk_long_text_multiple_chunks() -> None:
    text = ("Lorem ipsum dolor sit amet. " * 200).strip()
    chunks = chunker.chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)  # small slack for separator boundaries


def test_chunk_preserves_sinhala_unicode() -> None:
    text = "ආයුබෝවන්. මෙය සිංහල පාඨයකි. " * 30
    chunks = chunker.chunk_text(text, chunk_size=100)
    joined = "".join(chunks)
    assert "ආයුබෝවන්" in joined
    assert "සිංහල" in joined
