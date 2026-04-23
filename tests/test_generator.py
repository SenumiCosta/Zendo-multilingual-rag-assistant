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


def test_generate_answer_calls_inference_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeMessage:
        content = "42 is the answer."

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeClient:
        def chat_completion(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return FakeCompletion()

    generator._get_client.cache_clear()
    monkeypatch.setattr(generator, "_get_client", lambda _n, _p=None: FakeClient())

    out = generator.generate_answer("What is it?", ["context"], model_name="fake")
    assert out == "42 is the answer."
    # System + user message structure
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]
    assert "Question: What is it?" in captured["messages"][1]["content"]
