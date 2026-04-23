"""API endpoints for the RAG assistant.

Routes:
    GET  /health         -> simple liveness check
    POST /upload         -> ingest a PDF into the FAISS index
    POST /chat           -> answer a question using the current index
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.rag import pipeline
from backend.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)
router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "vector_db"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[list[tuple[str, str]]] = None
    k: int = 3


class Source(BaseModel):
    text: str
    score: float
    index: int

    @classmethod
    def from_result(cls, r: RetrievalResult) -> "Source":
        return cls(text=r.text, score=r.score, index=r.index)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    chunks: int


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        n = pipeline.ingest_pdf(dest, INDEX_DIR)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    log.info("ingested %s (%d chunks)", file.filename, n)
    return UploadResponse(filename=file.filename, chunks=n)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        answer, sources = pipeline.rag_pipeline(
            req.question,
            INDEX_DIR,
            k=req.k,
            history=req.history,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            409,
            "No documents indexed yet. Upload a PDF via /upload first.",
        ) from exc
    return ChatResponse(
        answer=answer,
        sources=[Source.from_result(r) for r in sources],
    )
