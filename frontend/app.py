"""Gradio UI for the Zendo RAG assistant.

Talks to the FastAPI backend over HTTP. Run the backend first:
    uvicorn backend.main:app --reload --port 8000

Then launch the UI:
    python frontend/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(600.0)


def upload_pdf(file_obj) -> str:
    if file_obj is None:
        return "No file selected."
    path = Path(file_obj.name)
    with path.open("rb") as fh, httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{BACKEND_URL}/upload",
            files={"file": (path.name, fh, "application/pdf")},
        )
    if resp.status_code != 200:
        return f"Upload failed: {resp.status_code} {resp.text}"
    data = resp.json()
    return f"Indexed {data['filename']}: {data['chunks']} chunks."


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
    payload = {"question": message, "history": tuple_history, "k": 3}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BACKEND_URL}/chat", json=payload)
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}"
    data = resp.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    if sources:
        src_block = "\n\n---\n**Sources:**\n" + "\n".join(
            f"- ({s['score']:.2f}) {s['text'][:160].strip()}..."
            for s in sources
        )
        return answer + src_block
    return answer


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Zendo Multilingual RAG") as demo:
        gr.Markdown(
            "# Zendo — Multilingual RAG Assistant\n"
            "Upload a PDF, then ask questions in **Sinhala or English**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Index PDF", variant="primary")
                upload_status = gr.Markdown()
                upload_btn.click(upload_pdf, inputs=file_in, outputs=upload_status)
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=chat_fn,
                    examples=[
                        "What is this document about?",
                        "මෙම ලේඛනය ගැන සාරාංශයක් දෙන්න.",
                    ],
                )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
