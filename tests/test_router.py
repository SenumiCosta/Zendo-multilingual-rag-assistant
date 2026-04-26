"""Tests for backend.rag.router."""

from __future__ import annotations

from backend.rag import router


def test_empty_query_routes_simple() -> None:
    r = router.route("")
    assert r.name == "simple"


def test_short_factual_is_simple() -> None:
    r = router.route("What is AI?")
    assert r.name == "simple"
    assert r.use_hyde is False
    assert r.use_rerank is False


def test_how_why_routes_complex() -> None:
    r = router.route("Why does gradient descent converge slowly?")
    assert r.name == "complex"
    assert r.use_hyde is True
    assert r.use_rerank is True


def test_long_query_routes_complex() -> None:
    # 13 words → complex via word-count gate
    q = " ".join(["term"] * 13)
    assert router.route(q).name == "complex"


def test_clause_reference_routes_relational() -> None:
    r = router.route("How does clause 5 connect to reference 7?")
    assert r.name == "relational"


def test_sinhala_how_routes_complex() -> None:
    r = router.route("AI කොහොමද වැඩ කරන්නේ?")
    assert r.name == "complex"
