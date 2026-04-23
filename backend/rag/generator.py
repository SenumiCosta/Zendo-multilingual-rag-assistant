"""LLM answer generation via HuggingFace Inference API.

Uses `huggingface_hub.InferenceClient.chat_completion` so we don't have
to download or run the LLM locally. Requires `HUGGINGFACE_TOKEN` in the
environment (a free "Read" token from huggingface.co/settings/tokens).

Falls back to a local transformers pipeline if the token is absent — use
that only for offline dev.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterable

log = logging.getLogger(__name__)

# Default to a widely-available, ungated, multilingual model on the HF
# Inference API. Override via MODEL_NAME in .env.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512

_SINHALA_RANGE = range(0x0D80, 0x0DFF + 1)


def detect_language(text: str) -> str:
    """Return 'si' if the text contains any Sinhala codepoints, else 'en'."""
    for ch in text:
        if ord(ch) in _SINHALA_RANGE:
            return "si"
    return "en"


def _system_prompt(language: str) -> str:
    lang_line = "Answer in Sinhala." if language == "si" else "Answer in English."
    return (
        "You are a helpful assistant answering questions from provided context.\n"
        "Rules:\n"
        "- Use ONLY the context below. Do not use outside knowledge.\n"
        "- If the answer is not in the context, say you don't know.\n"
        f"- {lang_line}"
    )


def _user_prompt(
    question: str,
    context_chunks: Iterable[str],
    history: list[tuple[str, str]] | None,
) -> str:
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    if not context:
        context = "(no context retrieved)"
    history_block = ""
    if history:
        turns = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
        history_block = f"\nConversation so far:\n{turns}\n"
    return (
        f"{history_block}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_prompt(
    question: str,
    context_chunks: Iterable[str],
    language: str = "auto",
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Compose a single-string prompt (kept for tests / local fallback)."""
    lang = detect_language(question) if language == "auto" else language
    return _system_prompt(lang) + "\n\n" + _user_prompt(question, context_chunks, history)


@lru_cache(maxsize=1)
def _get_client(model_name: str, provider: str | None = None):
    """Cached InferenceClient pointed at the chosen remote model.

    Set LLM_PROVIDER in .env (e.g. "hf-inference", "together", "fireworks-ai",
    "cerebras", "nebius") to force a specific Inference Provider when the
    default auto-routing picks one that doesn't serve the chosen model.
    """
    from huggingface_hub import InferenceClient

    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or None
    )
    if not token or token == "your_token_here":
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set. Get a free Read token at "
            "https://huggingface.co/settings/tokens and add it to .env."
        )
    kwargs: dict = {"model": model_name, "token": token}
    if provider:
        kwargs["provider"] = provider
    return InferenceClient(**kwargs)


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    """Local fallback pipeline — only used if the remote call fails."""
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )


def generate_answer(
    question: str,
    context_chunks: Iterable[str],
    language: str = "auto",
    history: list[tuple[str, str]] | None = None,
    model_name: str | None = None,
) -> str:
    """Call the HF Inference API and return the answer text."""
    lang = detect_language(question) if language == "auto" else language
    name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)

    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": _user_prompt(question, context_chunks, history)},
    ]

    provider = os.getenv("LLM_PROVIDER") or None
    client = _get_client(name, provider)
    log.info("calling HF Inference API: model=%s provider=%s", name, provider or "auto")
    try:
        completion = client.chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"LLM call failed for model '{name}'. "
            "This usually means the model isn't served on your HF plan. "
            "Pick a 'warm' model at "
            "https://huggingface.co/models?inference=warm&pipeline_tag=text-generation "
            "and set MODEL_NAME in .env. "
            f"Original error: {exc}"
        ) from exc
    return completion.choices[0].message.content.strip()
