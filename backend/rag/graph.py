"""GraphRAG — STUB.

Design target: extract entities + relationships from the corpus into a
knowledge graph (Neo4j), then answer relationship queries by traversing
the graph instead of doing pure vector search.

Current status: NOT IMPLEMENTED. `router.py` routes `relational` queries
to the standard hybrid+rerank path for now.

## Planned implementation

1. Entity / relation extraction
    - Microsoft's LazyGraphRAG (2025) for practical indexing cost.
    - Or simple spaCy-based NER for English + regex-driven Sinhala NER.

2. Graph store
    - Neo4j (docker-compose up for local dev) with schema:
      `(:Entity {name, type}) -[:RELATES {verb, chunk_id}]-> (:Entity)`

3. Retrieval
    - Cypher queries for n-hop traversals.
    - Seed entities from the user question, expand 1-2 hops, return
      paths + supporting chunks to the LLM.

4. Pipeline integration
    - `router.Route.name == "relational"` branches here.
    - Falls back to hybrid retrieval when no entities match.

## Why not built yet

- Requires Neo4j deployment (docker + network config).
- Entity extraction for Sinhala is an open problem; would need a custom
  model or translation-first pipeline.
- High effort (1 week+), low marginal gain for the v0.2 demo.

Tracked in `implementation.md` under the v0.2 roadmap.
"""

from __future__ import annotations

from typing import NoReturn


def extract_entities(_text: str) -> NoReturn:
    raise NotImplementedError(
        "GraphRAG is a v0.3 feature. See backend/rag/graph.py for the design."
    )


def graph_search(_query: str) -> NoReturn:
    raise NotImplementedError(
        "GraphRAG is a v0.3 feature. See backend/rag/graph.py for the design."
    )
