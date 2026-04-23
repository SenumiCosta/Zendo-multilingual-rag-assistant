# 4-Day Implementation Plan — Zendo Multilingual RAG Assistant

## Context

Multilingual (Sinhala + English) RAG Assistant for document Q&A. The repo is a scaffolded skeleton — directories, `requirements.txt`, and `.env.example` are in place, but all Python files are empty stubs. Goal: turn the scaffold into a working end-to-end RAG pipeline (PDF upload → chunk → embed → FAISS → retrieve → LLM → answer) with a Gradio UI in 4 working days.

**Stack:** FastAPI · sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) · FAISS · Gemma/LLaMA · Gradio.

## Files to Create / Modify

| File | Purpose |
|---|---|
| `backend/rag/loader.py` | PDF → text (use `pypdf`) |
| `backend/rag/chunker.py` | Text → chunks (use `langchain.text_splitter.RecursiveCharacterTextSplitter`) |
| `backend/rag/embedder.py` | Chunks → vectors (`sentence-transformers`, model from `.env`) |
| `backend/rag/retriever.py` | FAISS index build + similarity search |
| `backend/rag/generator.py` | LLM call with grounded RAG prompt |
| `backend/api/routes.py` | `/upload`, `/chat`, `/health` endpoints |
| `backend/main.py` | FastAPI app bootstrap + router mount |
| `frontend/app.py` | Gradio chat UI — file upload + chatbox |
| `tests/` | pytest unit tests for each rag module |
| `.env` | Local copy of `.env.example` with real HF token |

Reuse libraries already pinned in `requirements.txt`.

---

## Day 1 — Document Processing Pipeline (Loader + Chunker + Embedder)

**Goal:** Turn a PDF on disk into a list of embedded chunks ready to index.

- Set up Python venv, install `requirements.txt`, copy `.env.example` → `.env`, add HF token.
- `loader.py`: `load_pdf(path) -> str` using `pypdf.PdfReader`; strip whitespace, preserve Sinhala Unicode (UTF-8).
- `chunker.py`: `chunk_text(text, chunk_size=500, overlap=50) -> list[str]` using `RecursiveCharacterTextSplitter`. Tune for mixed Sinhala/English text.
- `embedder.py`: lazy-loaded `SentenceTransformer(os.getenv("EMBEDDING_MODEL"))` singleton, `embed(texts) -> np.ndarray`.
- Drop 1 English + 1 Sinhala PDF into `data/documents/`; run a notebook to verify: load → chunk → embed → print shape `(n_chunks, 384)`.
- Unit tests: `test_loader.py`, `test_chunker.py`, `test_embedder.py`.

**Deliverable:** embedding matrix produced from a real PDF, no errors.

---

## Day 2 — Vector DB + Retriever + LLM Generator

**Goal:** Given a question, retrieve top-k chunks and get an LLM answer.

- `retriever.py`:
  - `build_index(embeddings) -> faiss.IndexFlatL2`
  - `save_index` / `load_index` → `data/vector_db/`
  - `search(query_embedding, k=3)` + chunk-text mapping.
- `generator.py`:
  - Load Gemma/LLaMA via `transformers` pipeline (or HF Inference API if no GPU).
  - `generate_answer(question, context_chunks, language="auto")` with a **strict grounding prompt**: "Use ONLY the provided context. If not in context, say you don't know. Answer in the same language as the question."
  - Language detection: Unicode-range check for Sinhala (U+0D80–U+0DFF) vs ASCII.
- Wire into a `rag_pipeline(question)` helper.
- Ask 5 Sinhala + 5 English questions manually; confirm grounded answers.
- Unit tests: `test_retriever.py`, smoke test for generator (mock LLM).

**Deliverable:** end-to-end answer via the RAG pipeline from CLI/notebook.

---

## Day 3 — FastAPI Backend + Gradio Frontend

**Goal:** Expose the pipeline over HTTP and give the user a browser UI.

- `backend/main.py`: FastAPI app, include router, CORS for Gradio, startup hook loads embedder + FAISS index.
- `backend/api/routes.py`:
  - `POST /upload` — PDF via `python-multipart` → loader → chunker → embedder → FAISS upsert + persist.
  - `POST /chat` — `{question, history?}` → `{answer, sources}`.
  - `GET /health`.
- `frontend/app.py`: `gr.ChatInterface` + `gr.File` upload; call backend via `httpx`.
- Conversation memory: last N turns passed to the generator prompt so follow-ups ("eya use karanne kohomada?") work.
- Run concurrently: `uvicorn backend.main:app --reload` on :8000, `python frontend/app.py` on :7860.

**Deliverable:** browser demo — upload PDF, chat in Sinhala/English, see grounded answers with sources.

---

## Day 4 — Hardening, Evaluation, Documentation

**Goal:** Make it demo-ready and recruiter-ready.

- **Evaluation:** ~20 Q/A pairs (Si + En) in `notebooks/eval.ipynb`; measure retrieval hit-rate (correct chunk in top-k) + qualitative answer accuracy.
- **Robustness:** handle empty/corrupt PDFs, oversize uploads, missing HF token. Cap top-k + token budget. Add `logging`.
- **Performance:** benchmark embed/search/LLM latency. Consider smaller model or `IndexIVFFlat` if corpus grows.
- **UI polish:** show source snippets under each answer; add "Clear chat" button.
- **Docs:** update `README.md` with setup, usage, architecture diagram, known limitations (Sinhala tokenizer quirks, hallucination mitigation).
- **Stretch:** Dockerfile, or hybrid search (BM25 + vector).

**Deliverable:** tagged v0.1 — working demo, tests passing, README walkthrough, evaluation numbers.

---

## Verification

**Daily:** `pytest tests/` all green + day's deliverable script/notebook runs.

**End of Day 4:**
- `uvicorn backend.main:app` + `python frontend/app.py` start cleanly.
- Sinhala PDF → Sinhala question → grounded Sinhala answer with source.
- English PDF → English question → grounded English answer with source.
- Question NOT in docs → model says "I don't know" (grounding works).
- `pytest` exits 0.

---

## v0.2 Backlog (deferred — ship v0.1 first)

Captured research for a follow-up plan. Do not start these during Days 1-4.

**Cheap wins (candidates for a v0.1.1):**
- Swap embedder to BGE-M3 (`BAAI/bge-m3`) — better Sinhala, 8K context.
- RAGAS for structured eval (faithfulness / answer_relevancy / context_precision).
- HyDE: LLM generates a hypothetical answer, embed that instead of the raw Sinhala query.

**Bigger additions:**
- Hybrid retrieval (dense + BM25) + cross-encoder reranker (Qwen3-Reranker).
- Adaptive RAG: query router choosing vector / agentic / graph pipelines.
- Agentic RAG with LangGraph (planner → retriever loop → critic).
- Multimodal via Gemini Embedding 2 (image + PDF + audio in one space).

**Research tracks:**
- On-device / edge SLM fallback (Gemma 3n / Qwen3.5-0.8B) for offline use.
- GraphRAG (Microsoft impl / LazyGraphRAG) for Sri Lankan entity-relationship corpora.
