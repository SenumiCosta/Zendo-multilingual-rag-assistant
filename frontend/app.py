"""Gradio UI for the Zendo RAG assistant.

Talks to the FastAPI backend over HTTP. Run the backend first:
    uvicorn backend.main:app --reload --port 8000

Then launch the UI:
    python frontend/app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(600.0)


def _capabilities() -> dict:
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get(f"{BACKEND_URL}/capabilities")
        if resp.status_code == 200:
            return resp.json()
    except httpx.HTTPError:
        pass
    return {"multimodal": False, "agentic": False}


def upload_pdf(file_obj, multimodal_index: bool) -> str:
    if file_obj is None:
        return "No file selected."
    path = Path(file_obj.name)
    try:
        with path.open("rb") as fh, httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{BACKEND_URL}/upload",
                files={"file": (path.name, fh, "application/pdf")},
                data={"multimodal_index": "true" if multimodal_index else "false"},
            )
    except httpx.ConnectError:
        return f"Cannot reach backend at {BACKEND_URL}. Is it running?"
    except httpx.TimeoutException:
        return "Backend timed out while indexing. Check the server logs."
    if resp.status_code != 200:
        return f"Upload failed: {resp.status_code} {resp.text}"
    data = resp.json()
    msg = f"Indexed {data['filename']}: {data['chunks']} text chunks."
    if data.get("images_indexed"):
        msg += f" {data['images_indexed']} page images embedded (Gemini)."
    return msg


def search_image_by_text(query: str):
    if not query or not query.strip():
        return None, "Enter a query."
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{BACKEND_URL}/search-image-text",
                data={"query": query, "k": 5},
            )
    except httpx.HTTPError as exc:
        return None, f"Backend error: {exc}"
    if resp.status_code != 200:
        return None, f"Error {resp.status_code}: {resp.text}"
    hits = resp.json().get("hits", [])
    return _hits_to_gallery(hits)


def search_image_by_image(image_path):
    if not image_path:
        return None, "No image uploaded."
    try:
        with open(image_path, "rb") as fh, httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{BACKEND_URL}/search-image-image",
                files={"file": (Path(image_path).name, fh, "image/png")},
                data={"k": 5},
            )
    except httpx.HTTPError as exc:
        return None, f"Backend error: {exc}"
    if resp.status_code != 200:
        return None, f"Error {resp.status_code}: {resp.text}"
    hits = resp.json().get("hits", [])
    return _hits_to_gallery(hits)


def reset_index() -> str:
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.post(f"{BACKEND_URL}/reset-index")
    except httpx.HTTPError as exc:
        return f"Error: {exc}"
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}"
    removed = resp.json().get("removed", [])
    return f"Cleared index. Removed: {', '.join(removed) if removed else '(nothing to remove)'}."


def _hits_to_gallery(hits: list[dict]):
    if not hits:
        return None, "No matches."
    items = [
        (
            f"{BACKEND_URL}/image?path={quote(h['path'], safe='')}",
            f"{h['pdf']} p.{h['page']} ({h['score']:.2f})",
        )
        for h in hits
    ]
    caption = "\n".join(f"- {h['pdf']} page {h['page']} — score {h['score']:.3f}" for h in hits)
    return items, caption


def chat_fn(message: str, history) -> str:
    # Gradio may pass tuples or OpenAI-style dicts depending on version.
    tuple_history: list[tuple[str, str]] = []
    if history:
        if isinstance(history[0], dict):
            pending_user: str | None = None
            for msg in history:
                if msg.get("role") == "user":
                    pending_user = msg.get("content", "")
                elif msg.get("role") == "assistant" and pending_user is not None:
                    tuple_history.append((pending_user, msg.get("content", "")))
                    pending_user = None
        else:
            tuple_history = [tuple(pair) for pair in history]
    payload = {"question": message, "history": tuple_history}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BACKEND_URL}/chat", json=payload)
    except httpx.ConnectError:
        return f"Cannot reach backend at {BACKEND_URL}. Start it with `uvicorn backend.main:app --reload --port 8000`."
    except httpx.TimeoutException:
        return "Backend timed out. The server may be stuck — restart it and try again."
    if resp.status_code == 409:
        return "No documents indexed yet. Upload a PDF and click **Index PDF** first."
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}"
    data = resp.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    meta = data.get("meta") or {}

    parts = [answer]
    if meta:
        route_line = (
            f"_route: **{meta.get('route', '?')}**"
            f" · {meta.get('total_ms', '?')}ms total_"
        )
        if "hyde_ms" in meta:
            route_line += f" _(hyde {meta['hyde_ms']}ms"
            if "rerank_ms" in meta:
                route_line += f", rerank {meta['rerank_ms']}ms"
            route_line += ")_"
        parts.append("\n\n" + route_line)
    if sources:
        src_block = "\n\n---\n**Sources:**\n" + "\n".join(
            f"- ({s['score']:.3f}) {s['text'][:160].strip()}..."
            for s in sources
        )
        parts.append(src_block)
    return "".join(parts)


def build_ui() -> gr.Blocks:
    caps = _capabilities()
    multimodal_on = bool(caps.get("multimodal"))

    with gr.Blocks(title="Zendo Multilingual RAG") as demo:
        gr.Markdown(
            "# Zendo — Adaptive Multilingual RAG Assistant\n"
            "Upload a PDF, then ask questions in **Sinhala or English**."
            + (
                "\n\n_Multimodal search: **enabled** (Gemini Embedding 2)._"
                if multimodal_on
                else "\n\n_Multimodal search: disabled. Set `GOOGLE_API_KEY` in `.env` to enable image search._"
            )
        )
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Upload PDF", file_types=[".pdf"])
                    multimodal_chk = gr.Checkbox(
                        label="Also embed page images (multimodal)",
                        value=False,
                        interactive=multimodal_on,
                        info=None if multimodal_on else "Disabled — set GOOGLE_API_KEY to enable.",
                    )
                    upload_btn = gr.Button("Index PDF", variant="primary")
                    upload_status = gr.Markdown()
                    upload_btn.click(
                        upload_pdf,
                        inputs=[file_in, multimodal_chk],
                        outputs=upload_status,
                    )
                    reset_btn = gr.Button("Reset index (clear all PDFs)", variant="secondary")
                    reset_btn.click(reset_index, outputs=upload_status)
                with gr.Column(scale=2):
                    gr.ChatInterface(
                        fn=chat_fn,
                        examples=[
                            "What is this document about?",
                            "මෙම ලේඛනය ගැන සාරාංශයක් දෙන්න.",
                            "Why does the proposed method outperform the baselines?",
                        ],
                    )

        with gr.Tab("Image search", interactive=multimodal_on):
            gr.Markdown(
                "Find PDF pages whose images / diagrams match a text query "
                "or a reference image. Requires multimodal indexing on upload."
            )
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(label="Text query (Sinhala or English)")
                    text_btn = gr.Button("Search by text", variant="primary")
                with gr.Column():
                    img_query = gr.Image(label="Or upload a reference image", type="filepath")
                    img_btn = gr.Button("Search by image", variant="primary")
            gallery = gr.Gallery(label="Top matches", columns=3, height=400)
            caption_out = gr.Markdown()
            text_btn.click(search_image_by_text, inputs=text_query, outputs=[gallery, caption_out])
            img_btn.click(search_image_by_image, inputs=img_query, outputs=[gallery, caption_out])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
