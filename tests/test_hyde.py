"""Tests for backend.rag.hyde."""

from __future__ import annotations

import pytest

from backend.rag import hyde


def test_hyde_without_token_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hyde.hypothetical_document("What is AI?") == "What is AI?"


def test_hyde_with_placeholder_token_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "your_token_here")
    assert hyde.hypothetical_document("q") == "q"


def test_hyde_api_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_dummy")

    class Boom:
        def chat_completion(self, **_):
            raise RuntimeError("network down")

    monkeypatch.setattr(hyde, "InferenceClient", lambda **_: Boom())
    assert hyde.hypothetical_document("q") == "q"


def test_hyde_success_concatenates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_dummy")

    class FakeMsg:
        content = "Hypothetical answer body."

    class FakeChoice:
        message = FakeMsg()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeClient:
        def chat_completion(self, **_):
            return FakeCompletion()

    monkeypatch.setattr(hyde, "InferenceClient", lambda **_: FakeClient())
    out = hyde.hypothetical_document("What is AI?")
    assert "What is AI?" in out
    assert "Hypothetical answer body." in out
