"""
Phase 1 orchestrator entry point.

Call run_phase1() to execute the full discovery-and-scoring pipeline.
Returns Phase1Output with ranked candidates for Phase 2.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.common.schemas import Phase1Output, RankedCandidate, TopicInput
from src.orchestrators.phase1.graph import Phase1State, build_phase1_graph


def _initial_state(
    run_id: str,
    topics: list[TopicInput],
    since_date: str,
    user_id: str,
) -> Phase1State:
    return Phase1State(
        run_id=run_id,
        since_date=since_date,
        user_id=user_id,
        topics=topics,
        search_results={},
        heat_scores={},
        filtered_results={},
        scored_filtered_topics=[],
        selector_output=None,
        judge_output=None,
        ranked_candidates=[],
        errors=[],
        run_status="pending",
        aap_token=None,
        total_tokens_used=0,
    )


async def run_phase1(
    run_id: str,
    topics: list[TopicInput],
    since_date: str,
    user_id: str,
) -> Phase1Output:
    """
    Execute the Phase 1 pipeline and return a Phase1Output.

    Args:
        run_id:     Unique identifier for this run. Passed through all agent calls
                    and OTel spans.
        topics:     List of TopicInput items — topic_text is never logged.
        since_date: ISO 8601 date string; articles older than this are excluded.
        user_id:    Identity of the requesting user; embedded in the AAP token.

    Returns:
        Phase1Output with ranked_candidates, errors, run_status, and token usage.
    """
    if not since_date:
        since_date = datetime.now(timezone.utc).date().isoformat()

    graph = build_phase1_graph()
    result: Phase1State = await graph.ainvoke(
        _initial_state(run_id, topics, since_date, user_id)
    )

    judge_out = result.get("judge_output")
    judged_articles = judge_out.judged_articles if judge_out else []

    return Phase1Output(
        run_id=run_id,
        ranked_candidates=result.get("ranked_candidates", []),
        errors=result.get("errors", []),
        run_status=result.get("run_status", "unknown"),
        total_tokens_used=result.get("total_tokens_used", 0),
        judged_articles=judged_articles,
    )
