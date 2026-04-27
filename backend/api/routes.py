"""API endpoints for the RAG assistant.

Routes:
    GET  /health             -> simple liveness check
    GET  /capabilities       -> report which optional features are active
    POST /upload             -> ingest a PDF (text + optional image embeddings)
    POST /chat               -> answer a question using the current index
    POST /search-image-text  -> text query → matching PDF page images (multimodal)
    POST /search-image-image -> image query → matching PDF page images (multimodal)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.rag import multimodal, pipeline
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
    meta: dict = {}


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    images_indexed: int = 0


class ImageHitOut(BaseModel):
    path: str
    page: int
    pdf: str
    score: float


class ImageSearchResponse(BaseModel):
    hits: list[ImageHitOut]


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/capabilities")
def capabilities() -> dict:
    """Report which optional features the running backend supports."""
    return {
        "multimodal": multimodal.is_available(),
        "agentic": True,
    }


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    multimodal_index: bool = Form(False),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        n = await run_in_threadpool(pipeline.ingest_pdf, dest, INDEX_DIR)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    images_indexed = 0
    if multimodal_index:
        if not multimodal.is_available():
            raise HTTPException(
                422,
                "Multimodal indexing requested but GOOGLE_API_KEY is not set "
                "or google-generativeai is not installed.",
            )
        try:
            images_indexed = await run_in_threadpool(
                multimodal.ingest_pdf_images, dest, INDEX_DIR
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("multimodal indexing failed")
            raise HTTPException(500, f"Multimodal indexing failed: {exc}") from exc

    log.info(
        "ingested %s (%d text chunks, %d page images)",
        file.filename, n, images_indexed,
    )
    return UploadResponse(filename=file.filename, chunks=n, images_indexed=images_indexed)


@router.post("/search-image-text", response_model=ImageSearchResponse)
async def search_image_by_text(query: str = Form(...), k: int = Form(5)) -> ImageSearchResponse:
    if not multimodal.is_available():
        raise HTTPException(503, "Multimodal search unavailable. Set GOOGLE_API_KEY.")
    try:
        hits = await run_in_threadpool(
            multimodal.text_to_image_search, query, INDEX_DIR, k
        )
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return ImageSearchResponse(
        hits=[ImageHitOut(path=h.path, page=h.page, pdf=h.pdf, score=h.score) for h in hits]
    )


@router.post("/search-image-image", response_model=ImageSearchResponse)
async def search_image_by_image(
    file: UploadFile = File(...),
    k: int = Form(5),
) -> ImageSearchResponse:
    if not multimodal.is_available():
        raise HTTPException(503, "Multimodal search unavailable. Set GOOGLE_API_KEY.")
    suffix = Path(file.filename or "query.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        hits = await run_in_threadpool(
            multimodal.image_to_image_search, tmp_path, INDEX_DIR, k
        )
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return ImageSearchResponse(
        hits=[ImageHitOut(path=h.path, page=h.page, pdf=h.pdf, score=h.score) for h in hits]
    )


@router.get("/image")
def get_indexed_image(path: str) -> FileResponse:
    """Serve a rendered page image.

    `path` is relative to data/vector_db/ and MUST resolve to a file under
    the images/ subdirectory with an image suffix. Anything else (FAISS
    index, manifest JSON, files outside the dir) is rejected.
    """
    images_root = (INDEX_DIR / multimodal.IMAGE_DIR_NAME).resolve()
    safe = (INDEX_DIR / path).resolve()
    if not str(safe).startswith(str(images_root)):
        raise HTTPException(400, "path must be inside images/")
    if safe.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "only image files are served")
    if not safe.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(safe)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        answer, sources, meta = pipeline.rag_pipeline(
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
        meta=meta,
    )
