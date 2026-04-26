"""Tests for backend.rag.multimodal.

The Gemini API and pypdfium2 are mocked so tests stay fast and don't
require a Google API key or PDF rendering libraries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.rag import multimodal


# ---------- availability ----------

def test_is_available_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert multimodal.is_available() is False


def test_require_available_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        multimodal._require_available()


# ---------- index lifecycle ----------

def _stub_gemini(monkeypatch: pytest.MonkeyPatch, dim: int = 8) -> None:
    """Pretend Gemini is available + returns deterministic small vectors."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")

    # Bypass the import check inside _require_available.
    monkeypatch.setattr(multimodal, "_configure_gemini", lambda: None)
    monkeypatch.setattr(
        multimodal, "GEMINI_EMBED_DIM", dim, raising=False
    )

    def fake_text_embed(texts):
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(hash(("text", t)) & 0xFFFFFFFF)
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            out[i] = v
        return out

    def fake_image_embed(paths):
        out = np.empty((len(paths), dim), dtype=np.float32)
        for i, p in enumerate(paths):
            rng = np.random.default_rng(hash(("img", str(p))) & 0xFFFFFFFF)
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            out[i] = v
        return out

    monkeypatch.setattr(multimodal, "embed_text", fake_text_embed)
    monkeypatch.setattr(multimodal, "embed_image", fake_image_embed)

    # is_available() also imports google.generativeai — stub out via sys.modules
    import sys
    sys.modules.setdefault("google", type(sys)("google"))
    sys.modules.setdefault("google.generativeai", type(sys)("google.generativeai"))


def _stub_pdf_render(monkeypatch: pytest.MonkeyPatch, n_pages: int = 3) -> None:
    """Pretend render_pdf_pages writes N empty PNGs."""
    def fake_render(pdf_path, out_dir, dpi=150):
        target = Path(out_dir) / Path(pdf_path).stem
        target.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n_pages):
            p = target / f"page_{i + 1:03d}.png"
            p.write_bytes(b"fake")
            paths.append(p)
        return paths

    monkeypatch.setattr(multimodal, "render_pdf_pages", fake_render)


def test_ingest_pdf_images_creates_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gemini(monkeypatch)
    _stub_pdf_render(monkeypatch, n_pages=4)

    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    index_dir = tmp_path / "index"

    n = multimodal.ingest_pdf_images(fake_pdf, index_dir)
    assert n == 4

    vectors = np.load(index_dir / multimodal.IMAGE_VECTORS_FILE)
    manifest = json.loads(
        (index_dir / multimodal.IMAGE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert vectors.shape[0] == 4
    assert len(manifest) == 4
    assert manifest[0]["pdf"] == "doc.pdf"
    assert manifest[0]["page"] == 1


def test_ingest_appends_to_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gemini(monkeypatch)
    _stub_pdf_render(monkeypatch, n_pages=2)

    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"%PDF a")
    pdf_b = tmp_path / "b.pdf"
    pdf_b.write_bytes(b"%PDF b")
    index_dir = tmp_path / "index"

    multimodal.ingest_pdf_images(pdf_a, index_dir)
    multimodal.ingest_pdf_images(pdf_b, index_dir)

    vectors = np.load(index_dir / multimodal.IMAGE_VECTORS_FILE)
    assert vectors.shape[0] == 4  # 2 pages × 2 PDFs


def test_text_to_image_search_returns_ranked_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gemini(monkeypatch)
    _stub_pdf_render(monkeypatch, n_pages=3)

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF")
    index_dir = tmp_path / "index"
    multimodal.ingest_pdf_images(pdf, index_dir)

    hits = multimodal.text_to_image_search("query text", index_dir, k=2)
    assert len(hits) == 2
    # Hits should be in descending score order
    assert hits[0].score >= hits[1].score
    assert all(h.pdf == "manual.pdf" for h in hits)


def test_search_without_index_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gemini(monkeypatch)
    with pytest.raises(FileNotFoundError):
        multimodal.text_to_image_search("q", tmp_path / "nope")


def test_image_to_image_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gemini(monkeypatch)
    _stub_pdf_render(monkeypatch, n_pages=3)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    index_dir = tmp_path / "index"
    multimodal.ingest_pdf_images(pdf, index_dir)

    query_img = tmp_path / "query.png"
    query_img.write_bytes(b"fake")
    hits = multimodal.image_to_image_search(query_img, index_dir, k=3)
    assert len(hits) == 3
