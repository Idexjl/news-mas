"""
Phase 2 orchestrator entry point.

Call run_phase2() to execute the summarization and quality-gate pipeline
over the ranked candidates produced by Phase 1.
Returns DigestOutput with digest entries, tombstones, and run metrics.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

from src.common.schemas import DigestOutput, Phase1Output
from src.orchestrators.phase2.graph import (
    Phase2State,
    _CONFIDENCE_ORDER,
    build_phase2_graph,
)


def _initial_state(
    phase1_output: Phase1Output,
    run_id: str,
    user_id: str,
) -> Phase2State:
    return Phase2State(
        run_id=run_id,
        user_id=user_id,
        ranked_candidates=phase1_output.ranked_candidates,
        summaries={},
        reviewer_verdicts={},
        gate_decisions={},
        retry_counts={},
        budget_exhausted={},
        errors=[],
        total_tokens_used=0,
        digest_entries=[],
        tombstones=[],
        run_status="pending",
        aap_token=None,
        phase1_tokens_used=phase1_output.total_tokens_used,
    )


def _overall_confidence(digest_entries) -> Literal["HIGH", "MEDIUM", "LOW"]:
    if not digest_entries:
        return "LOW"
    scores = [_CONFIDENCE_ORDER.get(e.confidence or "LOW", 1) for e in digest_entries]
    avg = sum(scores) / len(scores)
    if avg >= 2.5:
        return "HIGH"
    if avg >= 1.5:
        return "MEDIUM"
    return "LOW"


async def run_phase2(
    phase1_output: Phase1Output,
    run_id: str,
    user_id: str,
) -> DigestOutput:
    """
    Execute the Phase 2 pipeline and return a DigestOutput.

    Args:
        phase1_output: Result from run_phase1(), carrying ranked_candidates
                       and token usage from Phase 1.
        run_id:        Shared run identifier (same as Phase 1).
        user_id:       Identity of the requesting user.

    Returns:
        DigestOutput with digest_entries, tombstones, overall_confidence,
        run_status, and combined token usage.
    """
    graph = build_phase2_graph()
    result: Phase2State = await graph.ainvoke(
        _initial_state(phase1_output, run_id, user_id)
    )

    digest_entries = result.get("digest_entries", [])
    tombstones = result.get("tombstones", [])
    phase2_tokens = result.get("total_tokens_used", 0)
    total_tokens = phase1_output.total_tokens_used + phase2_tokens

    return DigestOutput(
        run_id=run_id,
        user_id=user_id,
        generated_at=datetime.now(timezone.utc),
        digest_entries=digest_entries,
        tombstones=tombstones,
        total_topics=len(phase1_output.ranked_candidates),
        passed_count=len(digest_entries),
        tombstoned_count=len(tombstones),
        total_tokens_used=total_tokens,
        run_duration_ms=0,  # duration tracked per-node in finalize_phase2
        overall_confidence=_overall_confidence(digest_entries),
        run_status=result.get("run_status", "unknown"),
        errors=result.get("errors", []),
    )
