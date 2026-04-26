"""HyDE — Hypothetical Document Embeddings.

For low-resource languages like Sinhala, query embeddings can be a
weaker signal than document embeddings. HyDE (Gao et al., 2022) asks the
LLM to draft a hypothetical answer, then embeds that draft instead of
the raw query, producing a richer semantic signal that matches better
against the corpus.

Cost: one extra LLM call per query. Skip for simple factual queries
(see router.py for the gating logic).
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import InferenceClient

log = logging.getLogger(__name__)

HYDE_PROMPT = (
    "Write a brief hypothetical answer (2-3 sentences) to the following "
    "question, as if you were quoting a technical document. Do not add "
    "disclaimers or apologies — just the hypothetical answer text.\n\n"
    "Question: {question}\n\nHypothetical answer:"
)


def hypothetical_document(question: str, model_name: str | None = None) -> str:
    """Return an LLM-generated hypothetical answer for the question.

    Falls back to returning the original question on any error (HyDE is
    a retrieval optimization, not correctness-critical).
    """
    name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token or token == "your_token_here":
        log.info("HyDE: no HF token, returning original query")
        return question
    try:
        client = InferenceClient(model=name, token=token)
        completion = client.chat_completion(
            messages=[
                {"role": "system", "content": "You write concise hypothetical answers to questions."},
                {"role": "user", "content": HYDE_PROMPT.format(question=question)},
            ],
            max_tokens=128,
            temperature=0.2,
        )
        hypo = completion.choices[0].message.content.strip()
        # Concat original question so BM25 can still match rare terms.
        return f"{question}\n{hypo}"
    except Exception as exc:  # noqa: BLE001
        log.warning("HyDE generation failed (%s); returning original query", exc)
        return question
