"""Simple evaluation harness.

Measures retrieval hit-rate @k against a small labelled set. A hit = the
ground-truth chunk substring appears in any of the top-k retrieved
chunks. Keeps the harness local and dependency-free; swap for RAGAS
later (see v0.1.1 backlog).

Usage:
    python scripts/evaluate.py --eval eval/qa_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag import retriever  # noqa: E402

DEFAULT_EVAL = ROOT / "eval" / "qa_set.json"
DEFAULT_INDEX = ROOT / "data" / "vector_db"


def load_qa_set(path: Path) -> list[dict]:
    """Load a list of {question, expected_substring, lang} records."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(qa_set: list[dict], index_dir: Path, k: int = 3) -> dict:
    index, chunks = retriever.load_index(index_dir)
    hits = 0
    per_item: list[dict] = []
    for item in qa_set:
        results = retriever.search(item["question"], index, chunks, k=k)
        top_texts = " ".join(r.text for r in results)
        hit = item["expected_substring"] in top_texts
        hits += int(hit)
        per_item.append({
            "question": item["question"],
            "lang": item.get("lang", "?"),
            "hit@k": hit,
            "top_score": results[0].score if results else 0.0,
        })
    return {
        "k": k,
        "n": len(qa_set),
        "hit_rate": hits / len(qa_set) if qa_set else 0.0,
        "items": per_item,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    qa_set = load_qa_set(args.eval)
    report = evaluate(qa_set, args.index, k=args.k)

    print(f"Retrieval hit-rate @{report['k']}: "
          f"{report['hit_rate']:.2%}  ({report['n']} questions)")
    print("-" * 60)
    for item in report["items"]:
        mark = "✓" if item["hit@k"] else "✗"
        print(f"  {mark} [{item['lang']}] {item['question'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
