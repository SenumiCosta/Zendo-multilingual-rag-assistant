"""LLM answer generation.

Wraps a causal LM (default: google/gemma-2b-it) behind a `generate_answer`
function that composes a grounded RAG prompt: the model is instructed to
answer ONLY from the retrieved context, say "I don't know" when the
context is insufficient, and reply in the same language as the question.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Protocol

DEFAULT_MODEL = "google/gemma-2b-it"
MAX_NEW_TOKENS = 256

_SINHALA_RANGE = range(0x0D80, 0x0DFF + 1)


class _TextGenerator(Protocol):
    def __call__(self, prompt: str, **kwargs: object) -> list[dict]: ...


def detect_language(text: str) -> str:
    """Return 'si' if the text contains any Sinhala codepoints, else 'en'."""
    for ch in text:
        if ord(ch) in _SINHALA_RANGE:
            return "si"
    return "en"


def build_prompt(
    question: str,
    context_chunks: Iterable[str],
    language: str = "auto",
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Compose the grounded RAG prompt sent to the LLM."""
    lang = detect_language(question) if language == "auto" else language
    lang_instruction = (
        "Answer in Sinhala." if lang == "si" else "Answer in English."
    )

    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    if not context:
        context = "(no context retrieved)"

    history_block = ""
    if history:
        turns = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
        history_block = f"\nConversation so far:\n{turns}\n"

    return (
        "You are a helpful assistant answering questions from provided context.\n"
        "Rules:\n"
        "- Use ONLY the context below. Do not use outside knowledge.\n"
        '- If the answer is not in the context, say you don\'t know.\n'
        f"- {lang_instruction}\n"
        f"{history_block}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str) -> _TextGenerator:
    """Load the HF text-generation pipeline lazily (heavy import)."""
    from transformers import pipeline  # local import so tests can mock

    token = os.getenv("HUGGINGFACE_TOKEN") or None
    return pipeline(
        "text-generation",
        model=model_name,
        token=token,
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
    """Run the grounded prompt through the LLM and return the answer text."""
    prompt = build_prompt(question, context_chunks, language=language, history=history)
    name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    generator = _load_pipeline(name)
    outputs = generator(prompt)
    text = outputs[0].get("generated_text", "") if outputs else ""
    # Some HF pipelines echo the prompt back — strip it.
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()
