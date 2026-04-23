"""Tests for backend.rag.generator.

The real HF pipeline is too heavy to load in unit tests, so we mock
`_load_pipeline` and only assert prompt composition + post-processing.
"""

from __future__ import annotations

import pytest

from backend.rag import generator


@pytest.fixture(autouse=True)
def _clear_pipeline_cache() -> None:
    generator._load_pipeline.cache_clear()


def test_detect_language_sinhala() -> None:
    assert generator.detect_language("ආයුබෝවන්") == "si"


def test_detect_language_english() -> None:
    assert generator.detect_language("Hello world") == "en"


def test_build_prompt_includes_rules_and_context() -> None:
    prompt = generator.build_prompt(
        "What is AI?",
        ["AI is artificial intelligence.", "It mimics human reasoning."],
    )
    assert "Use ONLY the context below" in prompt
    assert "AI is artificial intelligence." in prompt
    assert "Question: What is AI?" in prompt
    assert "Answer in English." in prompt


def test_build_prompt_switches_to_sinhala() -> None:
    prompt = generator.build_prompt("AI මොකක්ද?", ["AI යනු..."])
    assert "Answer in Sinhala." in prompt


def test_build_prompt_includes_history() -> None:
    prompt = generator.build_prompt(
        "How do I use it?",
        ["context"],
        history=[("What is AI?", "AI is artificial intelligence.")],
    )
    assert "Conversation so far" in prompt
    assert "What is AI?" in prompt


def test_build_prompt_handles_empty_context() -> None:
    prompt = generator.build_prompt("Hi", [])
    assert "(no context retrieved)" in prompt


def test_generate_answer_strips_echoed_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_pipeline(prompt: str, **_kw):
        captured["prompt"] = prompt
        return [{"generated_text": prompt + "42 is the answer."}]

    monkeypatch.setattr(generator, "_load_pipeline", lambda _n: fake_pipeline)

    out = generator.generate_answer("What is it?", ["context"], model_name="fake")
    assert out == "42 is the answer."
    assert "Question: What is it?" in captured["prompt"]
