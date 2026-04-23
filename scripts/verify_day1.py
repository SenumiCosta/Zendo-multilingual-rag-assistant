"""Day 1 verification script.

Runs the full ingestion pipeline on every PDF in data/documents/ and
prints shapes so you can sanity-check that loading, chunking, and
embedding all work before moving on.

Usage:
    python scripts/verify_day1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `backend` importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag import chunker, embedder, loader  # noqa: E402

DOCS_DIR = ROOT / "data" / "documents"


def main() -> int:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}. Drop one in and rerun.")
        return 1

    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        text = loader.load_pdf(pdf)
        print(f"  loaded: {len(text):,} chars")

        chunks = chunker.chunk_text(text)
        print(f"  chunked: {len(chunks)} chunks")
        if chunks:
            preview = chunks[0][:80].replace("\n", " ")
            print(f"  sample : {preview!r}")

        vectors = embedder.embed(chunks)
        print(f"  embedded: shape={vectors.shape}, dtype={vectors.dtype}")

    print("\nDay 1 pipeline OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
