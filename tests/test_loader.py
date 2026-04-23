"""Tests for backend.rag.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from backend.rag import loader


def _write_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal PDF with the given page texts.

    pypdf can't synthesize text-bearing pages easily, so we create blank
    pages and monkeypatch in tests that need real text extraction.
    """
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


def test_load_pdf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_pdf(tmp_path / "nope.pdf")


def test_load_pdf_empty_pdf_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["blank"])
    with pytest.raises(ValueError):
        loader.load_pdf(pdf)


def test_load_pdf_joins_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "fake.pdf"
    _write_pdf(pdf, ["a", "b"])

    texts = iter(["Hello world", "ආයුබෝවන්"])

    def fake_extract(self: object) -> str:  # noqa: ARG001
        return next(texts)

    monkeypatch.setattr("pypdf._page.PageObject.extract_text", fake_extract)

    result = loader.load_pdf(pdf)
    assert "Hello world" in result
    assert "ආයුබෝවන්" in result
    assert "\n\n" in result  # pages joined with double newlines
