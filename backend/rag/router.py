"""Adaptive query router.

Decides which retrieval pipeline to use for a given query. We keep the
classifier rule-based so it's free, fast, and debuggable — a small LLM
classifier can replace these heuristics later (see v0.2 backlog).

Routes:
    - "simple"  : short factual question → vanilla hybrid retrieval (k=3)
    - "complex" : multi-hop / reasoning → HyDE + hybrid + rerank (k=5)
    - "relational" : entity-relationship → GraphRAG (stub; falls back to
      complex for now, see graph.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Multi-hop indicators — words suggesting the query chains or compares facts.
_COMPLEX_PATTERNS = [
    r"\b(how|why|explain|describe|compare|contrast|relationship|between|impact|effect|because|therefore|reason)\b",
    r"\b(step|steps|process|workflow|pipeline)\b",
    # Sinhala analogues
    r"කොහොමද|ඇයි|විස්තර|සාරාංශ|සංසන්දන|සම්බන්ධ",
]

# Relational indicators — entities + linking words.
_RELATIONAL_PATTERNS = [
    r"\b(connect|linked|related|reference|refers?\s+to|citation|cited)\b",
    r"\bclause\s+\d+",
    r"\breference\s+\d+",
]

_COMPLEX_RE = re.compile("|".join(_COMPLEX_PATTERNS), re.IGNORECASE | re.UNICODE)
_RELATIONAL_RE = re.compile("|".join(_RELATIONAL_PATTERNS), re.IGNORECASE | re.UNICODE)


@dataclass
class Route:
    name: str
    use_hyde: bool
    use_rerank: bool
    k: int
    reason: str


def route(query: str) -> Route:
    q = (query or "").strip()
    if not q:
        return Route("simple", use_hyde=False, use_rerank=False, k=3, reason="empty")

    if _RELATIONAL_RE.search(q):
        return Route(
            "relational",
            use_hyde=True,
            use_rerank=True,
            k=5,
            reason="matched relational pattern",
        )

    # Multi-clause / long questions are almost always 'complex'
    word_count = len(q.split())
    if word_count >= 12 or _COMPLEX_RE.search(q):
        return Route(
            "complex",
            use_hyde=True,
            use_rerank=True,
            k=5,
            reason=f"complex indicators (words={word_count})",
        )

    return Route(
        "simple",
        use_hyde=False,
        use_rerank=False,
        k=3,
        reason="short factual",
    )
