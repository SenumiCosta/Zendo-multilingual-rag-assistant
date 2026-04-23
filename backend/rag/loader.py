"""PDF -> text loader.

Reads a PDF from disk and returns its full text content as a single UTF-8
string. Designed to preserve Sinhala Unicode characters alongside English.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: str | Path) -> str:
    """Load a PDF file and return its text content.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text with pages joined by double newlines. Leading/trailing
        whitespace is stripped. Sinhala characters (U+0D80–U+0DFF) are
        preserved as-is.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)

    if not pages:
        raise ValueError(f"No extractable text in {pdf_path}")

    return "\n\n".join(pages)
