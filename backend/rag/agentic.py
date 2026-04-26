"""Agentic RAG — plan → retrieve → critic → (re-retrieve) → answer.

A LangGraph-based alternative to the linear pipeline. Used when the
router classifies a query as 'complex' AND agentic mode is enabled
(AGENTIC_MODE=1 in .env, or force_agentic=True on the API call).

The graph has four nodes:

    plan     : LLM reformulates the query into a search-friendly form
    retrieve : hybrid search → top-k candidates → optional rerank
    critic   : LLM scores how well the candidates cover the question
    answer   : LLM generates the final grounded response

If the critic score is below threshold AND we haven't re-retrieved yet,
the graph loops back to `plan` with the critic's feedback to refine the
search query. Otherwise it proceeds to `answer`.

Falls back to the linear pipeline if LangGraph fails to import or the
loop exceeds `MAX_ITERATIONS` without convergence.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TypedDict

from backend.rag import generator, hybrid, reranker
from backend.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)

MAX_ITERATIONS = 2
CRITIC_THRESHOLD = 0.6  # 0-1 scale, parsed from LLM output
CANDIDATE_POOL = 20


class AgenticState(TypedDict, total=False):
    question: str
    search_query: str
    iteration: int
    candidates: list[RetrievalResult]
    critic_score: float
    critic_feedback: str
    answer: str
    history: list[tuple[str, str]] | None
    timings: dict


# ---------- Nodes ----------

def plan_node(state: AgenticState, *, index) -> AgenticState:  # noqa: ANN001
    """Reformulate the question into a better search query."""
    question = state["question"]
    feedback = state.get("critic_feedback", "")
    iteration = state.get("iteration", 0)
    t0 = time.perf_counter()

    if iteration == 0:
        # First pass: use the question as-is (HyDE already ran upstream if enabled).
        search_query = question
    else:
        # Subsequent passes: ask the LLM to rewrite using the critic's feedback.
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite search queries. Given a question and feedback "
                    "about why a previous search failed, produce ONE better "
                    "search query (no explanations, just the query)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Previous retrieval feedback: {feedback}\n\n"
                    "Rewritten search query:"
                ),
            },
        ]
        search_query = _llm_short(prompt_messages, fallback=question)

    timings = state.get("timings", {})
    timings[f"plan_iter{iteration}_ms"] = int((time.perf_counter() - t0) * 1000)
    return {**state, "search_query": search_query, "timings": timings}


def retrieve_node(state: AgenticState, *, index) -> AgenticState:  # noqa: ANN001
    """Hybrid search + optional rerank for the current search query."""
    t0 = time.perf_counter()
    cands = hybrid.hybrid_search(
        index, state["search_query"], k=CANDIDATE_POOL, candidate_pool=CANDIDATE_POOL
    )
    cands = reranker.rerank(state["question"], cands, top_k=5)
    timings = state.get("timings", {})
    timings[f"retrieve_iter{state.get('iteration', 0)}_ms"] = int(
        (time.perf_counter() - t0) * 1000
    )
    return {**state, "candidates": cands, "timings": timings}


def critic_node(state: AgenticState, *, index) -> AgenticState:  # noqa: ANN001
    """LLM judges whether the retrieved chunks can answer the question."""
    t0 = time.perf_counter()
    question = state["question"]
    candidates = state.get("candidates", [])
    if not candidates:
        score, feedback = 0.0, "No chunks retrieved."
    else:
        context = "\n\n".join(f"[{i + 1}] {c.text[:400]}" for i, c in enumerate(candidates[:5]))
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval critic. On a 0.0-1.0 scale, rate how "
                    "well the provided context can answer the question. Reply "
                    "on exactly two lines:\n"
                    "Score: <0.0-1.0>\n"
                    "Feedback: <one short sentence about what's missing, or 'OK'>"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ]
        raw = _llm_short(prompt_messages, fallback="Score: 1.0\nFeedback: OK")
        score, feedback = _parse_critic(raw)

    iteration = state.get("iteration", 0) + 1
    timings = state.get("timings", {})
    timings[f"critic_iter{iteration - 1}_ms"] = int((time.perf_counter() - t0) * 1000)
    log.info("critic iter=%d score=%.2f feedback=%s", iteration, score, feedback)
    return {
        **state,
        "iteration": iteration,
        "critic_score": score,
        "critic_feedback": feedback,
        "timings": timings,
    }


def answer_node(state: AgenticState, *, index) -> AgenticState:  # noqa: ANN001
    """Produce the final grounded answer from the best candidates."""
    t0 = time.perf_counter()
    context = [c.text for c in state.get("candidates", [])]
    answer = generator.generate_answer(
        state["question"], context, history=state.get("history")
    )
    timings = state.get("timings", {})
    timings["answer_ms"] = int((time.perf_counter() - t0) * 1000)
    return {**state, "answer": answer, "timings": timings}


# ---------- Edge logic ----------

def should_loop(state: AgenticState) -> str:
    """Route: loop back to `plan` or proceed to `answer`."""
    if state.get("critic_score", 0.0) >= CRITIC_THRESHOLD:
        return "answer"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "answer"  # give up the loop, still answer with best candidates
    return "plan"


# ---------- Graph construction ----------

def build_graph(index: hybrid.HybridIndex):
    """Compile a LangGraph state machine bound to a specific hybrid index."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AgenticState)
    g.add_node("plan", lambda s: plan_node(s, index=index))
    g.add_node("retrieve", lambda s: retrieve_node(s, index=index))
    g.add_node("critic", lambda s: critic_node(s, index=index))
    g.add_node("answer", lambda s: answer_node(s, index=index))

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "critic")
    g.add_conditional_edges("critic", should_loop, {"plan": "plan", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


# ---------- Public entrypoint ----------

def agentic_answer(
    question: str,
    index_dir: str | Path,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[RetrievalResult], dict]:
    """Run the agentic pipeline. Returns (answer, sources, meta)."""
    idx = hybrid.load_hybrid_index(index_dir)
    graph = build_graph(idx)
    t0 = time.perf_counter()
    final: AgenticState = graph.invoke(  # type: ignore[assignment]
        {
            "question": question,
            "iteration": 0,
            "history": history,
            "timings": {},
        }
    )
    total_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "route": "agentic",
        "iterations": final.get("iteration", 1),
        "critic_score": round(final.get("critic_score", 0.0), 3),
        "total_ms": total_ms,
        **final.get("timings", {}),
    }
    return final.get("answer", ""), final.get("candidates", []), meta


# ---------- helpers ----------

def _llm_short(messages: list[dict], *, fallback: str) -> str:
    """Single short LLM call, catching every error."""
    try:
        client = generator._get_client(
            os.getenv("MODEL_NAME", generator.DEFAULT_MODEL),
            os.getenv("LLM_PROVIDER") or None,
        )
        completion = client.chat_completion(
            messages=messages,
            max_tokens=128,
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("agentic LLM helper failed (%s); using fallback", exc)
        return fallback


def _parse_critic(raw: str) -> tuple[float, str]:
    """Extract (score, feedback) from the critic's reply."""
    score = 0.0
    feedback = "parse failed"
    for line in raw.splitlines():
        low = line.strip().lower()
        if low.startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif low.startswith("feedback:"):
            feedback = line.split(":", 1)[1].strip()
    return max(0.0, min(1.0, score)), feedback
