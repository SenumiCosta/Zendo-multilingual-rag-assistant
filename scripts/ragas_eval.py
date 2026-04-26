"""RAGAS evaluation harness.

Runs the RAG pipeline against `eval/qa_set.json` and scores the outputs
with RAGAS metrics:

  - faithfulness     : is the answer grounded in the retrieved context?
  - answer_relevancy : does the answer address the question?
  - context_precision: are the top chunks the ones that actually help?

RAGAS calls an LLM (via LangChain) to judge. This script uses the same
HF Inference API configured in .env, so it should Just Work if your
chat pipeline works.

Usage:
    python scripts/ragas_eval.py --eval eval/qa_set.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag import pipeline  # noqa: E402

DEFAULT_EVAL = ROOT / "eval" / "qa_set.json"
DEFAULT_INDEX = ROOT / "data" / "vector_db"


def load_qa_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(qa_set: list[dict], index_dir: Path) -> list[dict]:
    """Run every question through the pipeline and capture answer + contexts."""
    rows = []
    for item in qa_set:
        q = item["question"]
        answer, sources, _ = pipeline.rag_pipeline(q, index_dir)
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": [s.text for s in sources],
            "ground_truth": item.get("ground_truth", item.get("expected_substring", "")),
        })
    return rows


def _score_with_ragas(rows: list[dict]) -> dict:
    """Run RAGAS metrics. Imports lazily so the script works without ragas installed."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "ragas_report.json")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Just collect pipeline outputs, don't score with RAGAS.")
    args = parser.parse_args()

    qa_set = load_qa_set(args.eval)
    print(f"Running pipeline on {len(qa_set)} questions...")
    rows = collect_runs(qa_set, args.index)

    report: dict = {"n": len(rows), "rows": rows}

    if not args.skip_ragas:
        if not os.getenv("HUGGINGFACE_TOKEN"):
            print("WARNING: HUGGINGFACE_TOKEN not set — RAGAS judge will fail.")
        try:
            print("Scoring with RAGAS (may take a minute, calls LLM per question)...")
            scores = _score_with_ragas(rows)
            report["ragas"] = scores
            print("\nScores:")
            for k, v in scores.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        except Exception as exc:  # noqa: BLE001
            print(f"RAGAS scoring failed: {exc}")
            report["ragas_error"] = str(exc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
