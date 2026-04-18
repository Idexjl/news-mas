from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

MAX_SUMMARY_RETRIES = 3


class Phase2State(TypedDict):
    run_id: str
    articles: list[dict]           # judged articles from Phase 1
    current_idx: int               # index of the article being processed
    pending_summary: Optional[dict]
    retry_count: int
    reviewed: list[dict]           # reviewer-approved summaries
    digest: list[dict]             # entries that passed the relevance gate
    error: Optional[str]


# ── Node stubs ────────────────────────────────────────────────────────────────
#
# [DPOP-TODO] All HTTP calls to agent endpoints from these nodes must use
# DPoP-bound Entra ID tokens. Use the same _call_agent() helper pattern as
# phase1/graph.py — see [DPOP-TODO] block there for the full template.
#
# Phase 2 token flow note:
#   - If this run is user-triggered, use entra.get_obo_token() to propagate
#     the user's identity through the summarizer → reviewer → relevance_gate
#     chain. The upstream user assertion is carried in state["user_assertion"].
#   - If this is a scheduled/batch run (no user context), use
#     get_client_credentials_token() as in Phase 1.
#
# [DPOP-TODO] Add user_assertion: str | None to Phase2State if OBO is needed:
#   class Phase2State(TypedDict):
#       ...
#       user_assertion: str | None   # populated by orchestrator entry point
#
# See DPOP_IMPLEMENTATION_GUIDE.md §3.3 (OBO vs client_credentials guidance).

async def summarizer_node(state: Phase2State) -> Phase2State:
    raise NotImplementedError


async def reviewer_node(state: Phase2State) -> Phase2State:
    raise NotImplementedError


async def relevance_gate_node(state: Phase2State) -> Phase2State:
    raise NotImplementedError


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_review(state: Phase2State) -> str:
    """
    If the reviewer rejected the summary and retries remain, send back to the
    summarizer. Otherwise advance to the relevance gate.
    """
    reviewer_approved = (state.get("pending_summary") or {}).get("approved", True)
    retries_exhausted = state.get("retry_count", 0) >= MAX_SUMMARY_RETRIES
    if reviewer_approved or retries_exhausted:
        return "relevance_gate"
    return "summarizer"


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build() -> StateGraph:
    g = StateGraph(Phase2State)
    g.add_node("summarizer", summarizer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("relevance_gate", relevance_gate_node)

    g.set_entry_point("summarizer")
    g.add_edge("summarizer", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "summarizer": "summarizer",
            "relevance_gate": "relevance_gate",
        },
    )
    g.add_edge("relevance_gate", END)
    return g


phase2_graph = _build().compile()
