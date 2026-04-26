# Zendo — Adaptive Multilingual RAG Assistant

**Adaptive Multilingual RAG for Sinhala-English Document Q&A — featuring
hybrid retrieval, cross-encoder reranking, HyDE query expansion, and
an adaptive query router.**

Upload a PDF, ask questions in Sinhala or English, get grounded answers
with citations and route-level observability.

## Architecture (v0.2)

```
                        User Query
                            │
                 ┌──────────▼──────────┐
                 │   Adaptive Router    │   ← router.py
                 └──┬──────┬────────┬───┘
      simple ───────┘      │        └─── relational / complex
          │                │                 │
          │                │    ┌────────────▼─────────────┐
          │                │    │  AGENTIC_MODE=1?         │
          │                │    │  yes → LangGraph loop   ─┼──┐
          │                │    │  no  → linear path      │  │
          │                │    └────────────┬─────────────┘  │
          │         ┌──────▼──────┐          │                │
          │         │    HyDE     │  ← hyde.py                │
          │         └──────┬──────┘          │                │
          ▼                ▼                 ▼                │
       ┌─────────────────────────────────────────┐            │
       │   Hybrid Retrieval  (hybrid.py)         │            │
       │   Dense (FAISS, BGE-M3) + BM25 → RRF    │            │
       └───────────────────┬─────────────────────┘            │
                           │  top-20 candidates               │
                  ┌────────▼────────┐                         │
                  │  Cross-Encoder  │  ← reranker.py          │
                  │    Reranker     │                         │
                  └────────┬────────┘                         │
                           │  top-3 / top-5                   │
                  ┌────────▼────────┐      ┌──────────────────▼──────────────────┐
                  │  LLM Generator  │      │  Agentic loop (agentic.py)          │
                  │  Grounded prompt│      │  plan → retrieve → critic           │
                  └────────┬────────┘      │     ↑________________|              │
                           │               │  critic < 0.6 → re-plan (≤2 iters)  │
                           │               │  critic ≥ 0.6 → answer              │
                           │               └──────────────────┬──────────────────┘
                           ▼                                  │
              Answer + Sources + Route meta  ◄────────────────┘
```

## Tech Stack

| Layer | Choice |
|---|---|
| Embeddings | `BAAI/bge-m3` (1024-dim, 8K ctx, 100+ languages) |
| Dense search | FAISS `IndexFlatIP` on normalized vectors |
| Sparse search | BM25Okapi (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| Query expansion | HyDE (hypothetical answer via same LLM) |
| Router | Rule-based (regex on query patterns + length) |
| LLM | HF Inference API (`Qwen/Qwen2.5-7B-Instruct` default) |
| Backend | FastAPI + `python-dotenv` |
| Frontend | Gradio `ChatInterface` + file upload |
| Eval | RAGAS (faithfulness / relevancy / context precision) + local hit-rate |

## Setup

```bash
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
copy .env.example .env             # fill in HUGGINGFACE_TOKEN
```

Get a free HF Read token: <https://huggingface.co/settings/tokens>.

## Run

```bash
# Terminal 1 — API
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 — UI
python frontend/app.py             # http://127.0.0.1:7860
```

## Usage

1. Upload PDF → click **Index PDF** → wait for "Indexed N chunks."
2. Ask in Sinhala or English.
3. Answer includes route metadata (e.g. `route: complex · 2140ms (hyde 650ms, rerank 180ms)`) and top sources.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok"}` |
| `POST` | `/upload` | multipart `file=<pdf>` | `{filename, chunks}` |
| `POST` | `/chat` | `{question, history?, k?}` | `{answer, sources[], meta}` |

OpenAPI docs: http://localhost:8000/docs

## Adaptive routing

`router.py` classifies each query into one of three routes:

| Route | Triggers | Linear Pipeline | Agentic Pipeline (`AGENTIC_MODE=1`) |
|---|---|---|---|
| `simple` | short factual questions (< 12 words, no how/why) | hybrid only, k=3 | (same — agentic opt-out) |
| `complex` | how/why/explain, long queries, multi-clause | HyDE → hybrid → rerank, k=5 | plan → retrieve → critic loop (≤2 iters) |
| `relational` | clause X / reference Y, "connects to", "cited" | HyDE → hybrid → rerank, k=5 | plan → retrieve → critic loop (≤2 iters) |

## Agentic mode

Set `AGENTIC_MODE=1` in `.env` to enable. Complex/relational queries then
run a LangGraph state machine:

1. **Plan** — reformulate query (uses critic feedback on re-runs).
2. **Retrieve** — hybrid search + rerank.
3. **Critic** — LLM scores retrieval coverage 0.0-1.0 + short feedback.
4. If score ≥ 0.6 OR iteration ≥ 2 → **Answer**. Otherwise loop back to Plan.

Costs 2-4 extra LLM calls per query. Designed for multi-hop questions where
a single retrieval round isn't enough (e.g. "how does clause 5 connect to
reference 7?"). Falls back to the linear path on any graph error.

## Testing

```bash
python -m pytest                   # fast suite (49 tests, all mocked)
python -m pytest -m slow           # includes real model downloads
```

## Evaluation

Two harnesses:

**Local hit-rate** (dependency-free):
```bash
python scripts/evaluate.py --eval eval/qa_set.json -k 3
```

**RAGAS** (LLM-judged):
```bash
python scripts/ragas_eval.py --eval eval/qa_set.json
# → eval/ragas_report.json with {faithfulness, answer_relevancy, context_precision}
```

Fill `eval/qa_set.json` with 10-20 Q/A pairs using the provided schema.

## Design Notes

- **Grounded prompt** explicitly forbids outside knowledge and asks for
  "I don't know" when context is insufficient.
- **Normalized embeddings + `IndexFlatIP`** → cosine similarity as dot product.
- **Reciprocal Rank Fusion** (k=60) fuses dense + BM25 without tuning score weights.
- **HyDE** concatenates the original query with the hypothetical answer so
  BM25 can still match rare terms (IDs, proper nouns).
- **Context capping** (`MAX_CONTEXT_CHARS=4000`) protects the LLM context window.
- **Sinhala tokenization**: regex treats U+0D80–U+0DFF as word characters for BM25.
- **Reranker is best-effort** — if the cross-encoder fails to load, RRF scores
  pass through (degraded precision, but pipeline still works).

## Known Limitations

- First query is slow: BGE-M3 + cross-encoder download ~2.5 GB total on first run.
- `Qwen/Qwen2.5-7B-Instruct` Sinhala quality is decent but not native. For
  better Sinhala, try `meta-llama/Llama-3.1-8B-Instruct` (gated — accept license).
- HF Inference API routes auto-pick a provider; if it fails with
  "not supported by any provider", set `LLM_PROVIDER=hf-inference` in `.env`
  or pick a different "warm" model.

## Multimodal search (optional)

When `GOOGLE_API_KEY` is set, the upload UI exposes a "multimodal" checkbox.
Tick it and every PDF page is rendered as a PNG and embedded with **Gemini
Embedding 2** (3072-dim, 100+ languages, all-modality).

This unlocks two new tabs:

- **Text → Image** — type a query in Sinhala or English, get the most
  relevant page images (charts, diagrams, photos) ranked by cosine
  similarity. Cross-modal AND cross-lingual.
- **Image → Image** — upload a reference picture; find pages with similar
  diagrams or figures inside the indexed corpus.

Get a free key: <https://aistudio.google.com/apikey>.

API endpoints:

| Method | Path | Body |
|---|---|---|
| `GET` | `/capabilities` | report which optional features are active |
| `POST` | `/search-image-text` | form: `query`, `k` |
| `POST` | `/search-image-image` | multipart `file=<png\|jpg>`, `k` |
| `GET` | `/image?path=...` | serve a rendered page image |

## Roadmap

See [`implementation.md`](./implementation.md). Remaining v0.3 stubs:

- `backend/rag/graph.py` — Microsoft GraphRAG / LazyGraphRAG (needs Neo4j).
- `backend/rag/edge.py` — on-device SLM fallback (deployment, not pure code).

Each documents the intended design without committing to the scope.

## License

See [`LICENSE`](./LICENSE).
