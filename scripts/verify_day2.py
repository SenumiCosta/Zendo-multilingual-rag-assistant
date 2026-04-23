"""Day 2 verification script.

Ingests every PDF in data/documents/ into a FAISS index at
data/vector_db/, then runs a couple of demo questions through the full
RAG pipeline and prints the answer + retrieved sources.

Usage:
    python scripts/verify_day2.py "your question here"
    python scripts/verify_day2.py             # uses default demo questions
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag import pipeline  # noqa: E402

DOCS_DIR = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "vector_db"

DEFAULT_QUESTIONS = [
    "What is this document about?",
    "මෙම ලේඛනය ගැන සාරාංශයක් දෙන්න.",
]


def main(argv: list[str]) -> int:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}. Drop one in and rerun.")
        return 1

    print(f"Ingesting {len(pdfs)} PDF(s) into {INDEX_DIR} ...")
    total = 0
    for pdf in pdfs:
        n = pipeline.ingest_pdf(pdf, INDEX_DIR)
        print(f"  {pdf.name}: {n} chunks")
        total += n
    print(f"Indexed {total} chunks total.\n")

    questions = argv[1:] or DEFAULT_QUESTIONS
    for q in questions:
        print(f"Q: {q}")
        answer, sources = pipeline.rag_pipeline(q, INDEX_DIR, k=3)
        print(f"A: {answer}\n")
        for i, src in enumerate(sources, 1):
            preview = src.text[:100].replace("\n", " ")
            print(f"  [{i}] score={src.score:.3f}  {preview!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
