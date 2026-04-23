# Zendo — Multilingual RAG Assistant

A Retrieval-Augmented Generation (RAG) system for **Sinhala + English** document Q&A.
Upload a PDF, ask questions in either language, get grounded answers with citations.

> **Pitch:** Adaptive Multilingual RAG for Sinhala-English Document Q&A — v0.1 ships a
> grounded baseline (dense FAISS + Gemma). v0.2 roadmap adds hybrid retrieval, agentic
> reasoning, multimodal search, and edge-device fallback. See [`implementation.md`](./implementation.md).

## Architecture (v0.1)

```
User → Gradio UI → FastAPI → Retriever (FAISS) → LLM (Gemma) → Response
                                    ↑
                    loader → chunker → embedder
```

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | FAISS (`IndexFlatIP`, normalized vectors = cosine) |
| LLM | `google/gemma-2b-it` (swap via `MODEL_NAME` env) |
| Frontend | Gradio |
| Eval | Local hit-rate harness (RAGAS in v0.1.1) |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # fill in HUGGINGFACE_TOKEN
```

For Gemma access, accept the license on
[huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
and set `HUGGINGFACE_TOKEN` in `.env`.

## Run

Two terminals:

```bash
# Terminal 1 — API
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — UI
python frontend/app.py            # opens http://localhost:7860
```

## Usage

1. Upload a PDF in the left panel → click **Index PDF**.
2. Ask a question in the chatbox (Sinhala or English).
3. Answer appears with ranked source snippets underneath.

## API

| Method | Path | Body / Returns |
|---|---|---|
| `GET` | `/health` | `{"status":"ok"}` |
| `POST` | `/upload` | multipart `file=<pdf>` → `{filename, chunks}` |
| `POST` | `/chat` | `{question, history?, k?}` → `{answer, sources[]}` |

OpenAPI docs at `http://localhost:8000/docs`.

## Tests

```bash
pytest                   # fast suite (uses mocks)
pytest -m slow           # includes real model downloads
```

## Evaluation

1. Ingest PDFs via the UI or `scripts/verify_day2.py`.
2. Edit `eval/qa_set.json` with question / expected-substring pairs.
3. `python scripts/evaluate.py --eval eval/qa_set.json -k 3`

Example output:

```
Retrieval hit-rate @3: 85.00%  (20 questions)
```

## Design Notes

- **Grounding:** The generator prompt forbids outside knowledge and requires
  "I don't know" when context is insufficient — mitigates hallucination.
- **Multilingual chunking:** Separator list in `chunker.py` includes the Sinhala
  danda (`। `) so sentence splits work in both languages.
- **Normalized embeddings + `IndexFlatIP`:** cosine similarity as a dot product.
- **Context capping:** `pipeline._cap_context` limits combined retrieved text
  to `MAX_CONTEXT_CHARS` to prevent LLM overflow.

## Known Limitations

- MiniLM is a 2022-era model; Sinhala coverage is adequate but not state-of-the-art.
  Swap to BGE-M3 on the v0.1.1 checklist (see [`implementation.md`](./implementation.md)).
- Gemma-2B first load downloads ~5 GB. Use a smaller model or HF Inference API
  for low-RAM machines.
- Vanilla dense retrieval only — no reranker or BM25 in v0.1.

## Roadmap

See [`implementation.md`](./implementation.md) for:
- The 4-day v0.1 build plan (already shipped).
- v0.1.1 cheap wins (BGE-M3, HyDE, RAGAS).
- v0.2 vision (hybrid retrieval, adaptive router, agentic LangGraph, multimodal, edge SLM).

## License

See [`LICENSE`](./LICENSE).
