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

## v0.2 — SHIPPED (2026-04-24)

Upgrades delivered on top of v0.1:

**Cheap wins (all done):**
- ✅ BGE-M3 embeddings (1024-dim, 8K ctx, stronger Sinhala) — `backend/rag/embedder.py`.
- ✅ RAGAS evaluation script — `scripts/ragas_eval.py`.
- ✅ HyDE query expansion — `backend/rag/hyde.py`.

**Bigger additions (all done):**
- ✅ Hybrid retrieval (dense FAISS + BM25 via RRF) — `backend/rag/hybrid.py`.
- ✅ Cross-encoder reranker (BGE-reranker-v2-m3) — `backend/rag/reranker.py`.
- ✅ Adaptive query router (rule-based) — `backend/rag/router.py`.
- ✅ Pipeline rewired to use router → HyDE → hybrid → rerank → LLM.
- ✅ API `/chat` returns `meta` (route, per-stage latencies).

**Agentic (done):**
- ✅ Agentic RAG with LangGraph — `backend/rag/agentic.py` (plan → retrieve →
  critic → re-plan loop, max 2 iterations, opt-in via `AGENTIC_MODE=1`).

**Multimodal (done — opt-in via `GOOGLE_API_KEY`):**
- ✅ PDF page rendering via `pypdfium2` (150 DPI PNGs).
- ✅ Gemini Embedding 2 client (text + image, 3072-dim, normalized).
- ✅ Persisted parallel image index (`image_vectors.npy` + `image_manifest.json`).
- ✅ `/search-image-text` and `/search-image-image` endpoints.
- ✅ "Image search" tab in the Gradio UI, conditionally enabled by
  `/capabilities`.

**Deferred to v0.3 (stubs in place):**
- ⏳ GraphRAG — `backend/rag/graph.py` stub. Needs Neo4j + entity extraction.
- ⏳ On-device / edge SLM — `backend/rag/edge.py` stub. Deployment problem,
  not a pure code task.

Each stub module raises `NotImplementedError` and documents the intended
design so it can be built without re-architecting.

---

## v0.2 Long-Form Roadmap — Adaptive Multimodal RAG (8 weeks, post-v0.1)

**Tagline:** "Adaptive Multimodal RAG for Sinhala-English Document Q&A — featuring hybrid retrieval, agentic reasoning, and on-device fallback."

**Target architecture:**
```
User Query (Sinhala / English / Image)
  → Query Router (Adaptive RAG)
      ├─ Simple        → Vector RAG (fast)
      ├─ Complex       → Agentic RAG (LangGraph loops)
      └─ Relationship  → GraphRAG
  → Hybrid Retrieval (Dense BGE-M3 + BM25)
  → Reranker (Qwen3-Reranker)
  → LLM (Cloud Gemma/LLaMA OR Edge SLM)
  → Grounded answer + citations
```

**Tech stack deltas from v0.1:**
| Layer | v0.1 (current) | v0.2 (target) |
|---|---|---|
| Embeddings | paraphrase-multilingual-MiniLM | BGE-M3 + Gemini Embedding 2 (multimodal) |
| Retrieval | Dense only (FAISS IP) | Hybrid (dense + BM25) + Qwen3-Reranker |
| Orchestration | Linear pipeline | LangGraph agentic loops |
| LLM | Cloud Gemma/LLaMA | Cloud + Edge (Gemma 3n / Qwen3.5-0.8B) fallback |
| Eval | Ad-hoc | RAGAS (faithfulness, answer_relevancy, context_precision) |

**Phased build:**
- **Phase 1 (Weeks 1-2)** — Foundation upgrade: swap to BGE-M3, add BM25 + hybrid retrieval.
- **Phase 2 (Weeks 3-4)** — Multimodal: Gemini Embedding 2, image-in-PDF extraction, Qwen reranker.
- **Phase 3 (Weeks 5-6)** — Adaptive intelligence: query router + LangGraph agentic flow (plan → retrieve → critique → re-retrieve).
- **Phase 4 (Week 7)** — Edge mode: on-device Gemma 3n / Qwen3.5-0.8B fallback for offline use.
- **Phase 5 (Week 8)** — Polish: RAGAS evaluation dashboard, demo video, deployment.

**Key differentiators vs "another RAG chatbot":**
1. Multimodal Sinhala ↔ English cross-modal search (Sinhala query → English doc image match).
2. Adaptive query routing — one system, three retrieval strategies.
3. Agentic reasoning loops (not linear retrieve→generate).
4. Edge fallback for unreliable-internet contexts (rural Sri Lanka).
5. Production eval with RAGAS, not "it works!" demos.

**Scope discipline:** Do NOT start any of this until v0.1 is shipped, tagged, and documented. This section is a backlog, not a todo list.
