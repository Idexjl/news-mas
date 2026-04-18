from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph


class Phase1State(TypedDict):
    run_id: str
    topics: list[str]
    raw_articles: list[dict]
    scored_articles: list[dict]
    filtered_articles: list[dict]
    selected_articles: list[dict]
    judged_articles: list[dict]
    error: Optional[str]


# ── Node stubs ────────────────────────────────────────────────────────────────
#
# [DPOP-TODO] All HTTP calls to agent endpoints from these nodes must use
# DPoP-bound Entra ID tokens. Per-request pattern:
#
#   from src.common.auth.agent_identity import load_agent_identity
#   from src.common.auth.entra import get_client_credentials_token
#   from src.common.auth.dpop import generate_dpop_proof
#
#   _identity = load_agent_identity("phase1_orchestrator")
#
#   async def _call_agent(url: str, payload: dict) -> dict:
#       token_resp = await get_client_credentials_token(
#           tenant_id=_identity.tenant_id,
#           client_id=_identity.client_id,
#           client_secret=_identity.client_secret,
#           scope=f"api://{target_client_id}/.default",
#           dpop_private_key=_identity.dpop_key,
#       )
#       access_token = token_resp["access_token"]
#       # Fresh proof per request — NEVER cache or reuse a proof
#       proof = generate_dpop_proof(
#           method="POST", uri=url,
#           private_key=_identity.dpop_key,
#           access_token=access_token,
#       )
#       async with httpx.AsyncClient() as client:
#           resp = await client.post(
#               url, json=payload,
#               headers={
#                   "Authorization": f"DPoP {access_token}",
#                   "DPoP": proof,
#               },
#           )
#           resp.raise_for_status()
#           return resp.json()
#
# Phase 1 uses client_credentials (no user context — scheduled batch job).
# See DPOP_IMPLEMENTATION_GUIDE.md §3.3 for when OBO applies instead.

async def search_node(state: Phase1State) -> Phase1State:
    raise NotImplementedError


async def heat_score_node(state: Phase1State) -> Phase1State:
    raise NotImplementedError


async def filter_node(state: Phase1State) -> Phase1State:
    raise NotImplementedError


async def selector_node(state: Phase1State) -> Phase1State:
    raise NotImplementedError


async def phase1_judge_node(state: Phase1State) -> Phase1State:
    raise NotImplementedError


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build() -> StateGraph:
    g = StateGraph(Phase1State)
    g.add_node("search", search_node)
    g.add_node("heat_score", heat_score_node)
    g.add_node("filter", filter_node)
    g.add_node("selector", selector_node)
    g.add_node("phase1_judge", phase1_judge_node)

    g.set_entry_point("search")
    g.add_edge("search", "heat_score")
    g.add_edge("heat_score", "filter")
    g.add_edge("filter", "selector")
    g.add_edge("selector", "phase1_judge")
    g.add_edge("phase1_judge", END)
    return g


phase1_graph = _build().compile()
