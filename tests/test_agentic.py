"""Tests for backend.rag.agentic.

The real LLM and reranker are mocked so tests run in milliseconds and
don't need network. Focus: critic parsing, loop termination, and the
happy-path through the graph.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.rag import agentic, embedder, hybrid, reranker
from backend.rag.retriever import RetrievalResult


# ---------- unit tests on helpers ----------

def test_parse_critic_valid() -> None:
    score, feedback = agentic._parse_critic("Score: 0.85\nFeedback: OK")
    assert score == 0.85
    assert feedback == "OK"


def test_parse_critic_clamps_out_of_range() -> None:
    assert agentic._parse_critic("Score: 2.0\nFeedback: x")[0] == 1.0
    assert agentic._parse_critic("Score: -0.3\nFeedback: x")[0] == 0.0


def test_parse_critic_garbage_defaults_to_zero() -> None:
    score, feedback = agentic._parse_critic("nonsense reply")
    assert score == 0.0
    assert "parse failed" in feedback


def test_should_loop_high_score_goes_to_answer() -> None:
    assert agentic.should_loop({"critic_score": 0.9, "iteration": 1}) == "answer"


def test_should_loop_low_score_loops_back() -> None:
    assert agentic.should_loop({"critic_score": 0.3, "iteration": 1}) == "plan"


def test_should_loop_max_iterations_forces_answer() -> None:
    assert (
        agentic.should_loop({"critic_score": 0.1, "iteration": agentic.MAX_ITERATIONS})
        == "answer"
    )


# ---------- integration: full graph with mocks ----------

@pytest.fixture
def _stub_embedder(monkeypatch: pytest.MonkeyPatch):
    class Fake:
        def encode(self, texts, **_kw):
            vectors = np.empty((len(texts), 4), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(hash(t) & 0xFFFFFFFF)
                v = rng.standard_normal(4).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-9
                vectors[i] = v
            return vectors

    embedder.get_model.cache_clear()
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda _n: Fake())


@pytest.fixture
def _stub_reranker(monkeypatch: pytest.MonkeyPatch):
    # Pass-through reranker (keeps input order, truncates to top_k).
    monkeypatch.setattr(
        reranker,
        "rerank",
        lambda _q, cands, top_k=5, **_kw: cands[:top_k],
    )


@pytest.fixture
def _stub_llm_helpers(monkeypatch: pytest.MonkeyPatch):
    """Mock the agentic LLM helper + the final generator call."""
    call_log: list[str] = []

    def fake_llm_short(messages, *, fallback):
        # Inspect the system prompt to decide which node called us.
        sys_prompt = messages[0]["content"].lower()
        if "retrieval critic" in sys_prompt:
            call_log.append("critic")
            return "Score: 0.9\nFeedback: OK"
        if "rewrite search queries" in sys_prompt:
            call_log.append("plan")
            return "rewritten query"
        call_log.append("unknown")
        return fallback

    monkeypatch.setattr(agentic, "_llm_short", fake_llm_short)
    monkeypatch.setattr(
        agentic.generator,
        "generate_answer",
        lambda q, ctx, history=None: f"ANSWER({len(ctx)} chunks)",
    )
    return call_log


def test_agentic_happy_path_one_iteration(
    tmp_path: Path, _stub_embedder, _stub_reranker, _stub_llm_helpers
) -> None:
    # Build a real hybrid index on disk so agentic_answer can load it.
    chunks = ["apple pie recipe", "quantum physics notes", "gardening tips"]
    vectors = embedder.embed(chunks)
    idx = hybrid.build_hybrid_index(chunks, vectors)
    hybrid.save_hybrid_index(idx, tmp_path)

    answer, sources, meta = agentic.agentic_answer("What is an apple pie?", tmp_path)

    assert answer.startswith("ANSWER(")
    assert len(sources) > 0
    assert meta["route"] == "agentic"
    assert meta["iterations"] == 1  # critic scored 0.9 → no loop
    assert meta["critic_score"] == 0.9


def test_agentic_loops_when_critic_scores_low(
    tmp_path: Path, _stub_embedder, _stub_reranker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critic returns low score first time, high the second → 2 iterations."""
    chunks = ["alpha content", "beta content"]
    vectors = embedder.embed(chunks)
    idx = hybrid.build_hybrid_index(chunks, vectors)
    hybrid.save_hybrid_index(idx, tmp_path)

    critic_scores = iter(["Score: 0.2\nFeedback: missing details", "Score: 0.9\nFeedback: OK"])

    def fake_llm_short(messages, *, fallback):
        sys_prompt = messages[0]["content"].lower()
        if "retrieval critic" in sys_prompt:
            return next(critic_scores)
        if "rewrite search queries" in sys_prompt:
            return "alpha refined"
        return fallback

    monkeypatch.setattr(agentic, "_llm_short", fake_llm_short)
    monkeypatch.setattr(
        agentic.generator, "generate_answer", lambda q, ctx, history=None: "final"
    )

    _, _, meta = agentic.agentic_answer("alpha?", tmp_path)
    assert meta["iterations"] == 2
    assert meta["critic_score"] == 0.9
