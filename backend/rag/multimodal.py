"""Multimodal search — Gemini Embedding 2.

Embeds PDF pages (rendered as images) into a shared 3072-dim vector space
with text queries, so a Sinhala/English text query can match a chart or
diagram inside an English PDF.

Activates only when GOOGLE_API_KEY is set in the environment. Without it,
all entry points raise a helpful error and the rest of the pipeline keeps
working text-only.

Why render pages as images instead of extracting embedded images?
PDFs from papers / manuals frequently have meaningful figures composed of
text + vector graphics, where the embedded raster ("/Image" object) is
just one tile. Rendering each page is uniform, lossless for diagrams, and
works on every PDF.

## Index layout (under data/vector_db/)
    images/<pdf_stem>/page_<n>.png    rendered page images
    image_vectors.npy                  (n, 3072) float32 matrix
    image_manifest.json                [{path, page, pdf} ...] aligned with vectors
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Gemini Embedding 2 (March 2026): all-modality, 3072-dim, 100+ languages.
GEMINI_EMBED_MODEL = "models/gemini-embedding-2"
GEMINI_EMBED_DIM = 3072

IMAGE_DIR_NAME = "images"
IMAGE_VECTORS_FILE = "image_vectors.npy"
IMAGE_MANIFEST_FILE = "image_manifest.json"

# Render resolution. 150 DPI is a good readability/size tradeoff.
PAGE_RENDER_DPI = 150


@dataclass
class ImageHit:
    path: str
    page: int
    pdf: str
    score: float


# ---------- public availability check ----------

def is_available() -> bool:
    """Return True if the Gemini API key is set AND google-generativeai is importable."""
    if not os.getenv("GOOGLE_API_KEY"):
        return False
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        return False
    return True


def _require_available() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Multimodal search requires GOOGLE_API_KEY in .env. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    try:
        import google.generativeai  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        ) from exc


# ---------- PDF page rendering ----------

def render_pdf_pages(pdf_path: str | Path, out_dir: str | Path, dpi: int = PAGE_RENDER_DPI) -> list[Path]:
    """Render every page of `pdf_path` to PNG under `out_dir/<pdf_stem>/`.

    Returns the list of written file paths in page order.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 not installed. Run: pip install pypdfium2 pillow"
        ) from exc

    pdf_path = Path(pdf_path)
    target_dir = Path(out_dir) / pdf_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72  # PDFium uses 72 DPI as the base
    written: list[Path] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            out = target_dir / f"page_{i + 1:03d}.png"
            pil.save(out, format="PNG", optimize=True)
            written.append(out)
    finally:
        pdf.close()
    log.info("rendered %d pages from %s", len(written), pdf_path.name)
    return written


# ---------- Gemini embedding wrappers ----------

@lru_cache(maxsize=1)
def _configure_gemini() -> None:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def embed_text(texts: list[str]) -> np.ndarray:
    """Embed text via Gemini Embedding 2. Returns (n, 3072) float32."""
    _require_available()
    if not texts:
        return np.empty((0, GEMINI_EMBED_DIM), dtype=np.float32)
    import google.generativeai as genai

    _configure_gemini()
    vectors: list[list[float]] = []
    # Gemini batches up to 100 inputs per request in current SDK.
    for chunk_start in range(0, len(texts), 100):
        batch = texts[chunk_start : chunk_start + 100]
        result = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        vectors.extend(result["embedding"])
    arr = np.array(vectors, dtype=np.float32)
    # Normalize for cosine-as-dot-product, matching our FAISS IP setup.
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def embed_image(image_paths: list[str | Path]) -> np.ndarray:
    """Embed images via Gemini Embedding 2. Returns (n, 3072) float32."""
    _require_available()
    if not image_paths:
        return np.empty((0, GEMINI_EMBED_DIM), dtype=np.float32)
    import google.generativeai as genai
    from PIL import Image

    _configure_gemini()
    vectors: list[list[float]] = []
    for path in image_paths:
        img = Image.open(path)
        result = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=img,
            task_type="retrieval_document",
        )
        vectors.append(result["embedding"])
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


# ---------- index lifecycle ----------

def ingest_pdf_images(pdf_path: str | Path, index_dir: str | Path) -> int:
    """Render PDF pages, embed them, append to the multimodal index.

    Returns the number of pages added.
    """
    _require_available()
    pdf_path = Path(pdf_path)
    index_dir = Path(index_dir)
    image_root = index_dir / IMAGE_DIR_NAME
    image_root.mkdir(parents=True, exist_ok=True)

    pages = render_pdf_pages(pdf_path, image_root)
    new_vectors = embed_image(pages)

    manifest_entries = [
        {
            # Always store POSIX-style relative paths so URLs are portable
            # across Windows / *nix.
            "path": p.relative_to(index_dir).as_posix(),
            "page": i + 1,
            "pdf": pdf_path.name,
        }
        for i, p in enumerate(pages)
    ]

    # Append to existing index if present.
    vectors_file = index_dir / IMAGE_VECTORS_FILE
    manifest_file = index_dir / IMAGE_MANIFEST_FILE
    if vectors_file.exists() and manifest_file.exists():
        existing = np.load(vectors_file)
        merged = np.concatenate([existing, new_vectors], axis=0)
        existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        merged_manifest = existing_manifest + manifest_entries
    else:
        merged = new_vectors
        merged_manifest = manifest_entries

    np.save(vectors_file, merged)
    manifest_file.write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("multimodal: added %d page images (total now %d)", len(pages), len(merged))
    return len(pages)


def _load_index(index_dir: Path) -> tuple[np.ndarray, list[dict]]:
    vectors_file = index_dir / IMAGE_VECTORS_FILE
    manifest_file = index_dir / IMAGE_MANIFEST_FILE
    if not vectors_file.exists() or not manifest_file.exists():
        raise FileNotFoundError(
            f"No multimodal index in {index_dir}. Index a PDF with multimodal=True first."
        )
    vectors = np.load(vectors_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return vectors, manifest


# ---------- search ----------

def text_to_image_search(query: str, index_dir: str | Path, k: int = 5) -> list[ImageHit]:
    """Embed a text query and rank PDF page images by cosine similarity."""
    _require_available()
    vectors, manifest = _load_index(Path(index_dir))
    q_vec = embed_text([query])[0]
    scores = vectors @ q_vec  # cosine since both sides are L2-normalized
    top = np.argsort(scores)[::-1][:k]
    return [
        ImageHit(
            path=manifest[i]["path"],
            page=manifest[i]["page"],
            pdf=manifest[i]["pdf"],
            score=float(scores[i]),
        )
        for i in top
    ]


def image_to_image_search(image_path: str | Path, index_dir: str | Path, k: int = 5) -> list[ImageHit]:
    """Embed an uploaded image and find the most similar pages in the index."""
    _require_available()
    vectors, manifest = _load_index(Path(index_dir))
    q_vec = embed_image([image_path])[0]
    scores = vectors @ q_vec
    top = np.argsort(scores)[::-1][:k]
    return [
        ImageHit(
            path=manifest[i]["path"],
            page=manifest[i]["page"],
            pdf=manifest[i]["pdf"],
            score=float(scores[i]),
        )
        for i in top
    ]
