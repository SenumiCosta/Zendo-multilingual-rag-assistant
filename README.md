# Zendo-multilingual-rag-assistant
Multilingual (Sinhala + English) AI Assistant with RAG for document Q&amp;A

# Multilingual RAG Assistant

A Retrieval-Augmented Generation (RAG) system supporting **Sinhala** and **English** 
for document-based question answering.

## Architecture

User → Frontend (Gradio) → Backend (FastAPI) → Retriever (FAISS) → LLM → Response

## Tech Stack

- **Backend**: FastAPI
- **Embeddings**: sentence-transformers (multilingual)
- **Vector DB**: FAISS
- **LLM**: Gemma / LLaMA
- **Frontend**: Gradio

## Status

🚧 Work in progress

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```