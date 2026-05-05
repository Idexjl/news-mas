"""
Full pipeline runner — chains Phase 1 and Phase 2 into a single call.

Use run_full_pipeline() for end-to-end execution from raw topics to digest.
Use run_phase1() / run_phase2() directly when you need to run phases separately
or resume from a Phase 1 result.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from src.common.observability import configure_langsmith, setup_telemetry
from src.common.schemas import DigestOutput, DigestTombstone, TopicInput
from src.orchestrators.phase1.runner import run_phase1
from src.orchestrators.phase2.runner import run_phase2


def _generate_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def _empty_digest(run_id: str, user_id: str, run_status: str = "no_candidates") -> DigestOutput:
    return DigestOutput(
        run_id=run_id,
        user_id=user_id,
        generated_at=datetime.now(timezone.utc),
        digest_entries=[],
        tombstones=[],
        total_topics=0,
        passed_count=0,
        tombstoned_count=0,
        total_tokens_used=0,
        run_duration_ms=0,
        overall_confidence="LOW",
        run_status=run_status,
        errors=[],
    )


async def run_full_pipeline(
    topics: list[TopicInput],
    since_date: str,
    user_id: str,
    *,
    run_id: str | None = None,
) -> DigestOutput:
    """
    Execute the complete Phase 1 + Phase 2 pipeline.

    Args:
        topics:     User-defined topics. topic_text is never logged.
        since_date: ISO 8601 date string; articles older than this are excluded.
        user_id:    Identity of the requesting user.
        run_id:     Optional run identifier. Generated if not provided.

    Returns:
        DigestOutput from Phase 2, or an empty DigestOutput if Phase 1 produced
        no candidates (e.g. no_topics, circuit_breaker_active).
    """
    configure_langsmith()
    setup_telemetry("news-mas-pipeline")

    if not since_date:
        since_date = datetime.now(timezone.utc).date().isoformat()

    rid = run_id or _generate_run_id()

    # ── Phase 1: discovery and scoring ─────────────────────────────────────────
    phase1 = await run_phase1(
        run_id=rid,
        topics=topics,
        since_date=since_date,
        user_id=user_id,
    )

    if not phase1.ranked_candidates:
        return _empty_digest(rid, user_id, phase1.run_status)

    # ── Phase 2: summarisation and quality gate ─────────────────────────────────
    return await run_phase2(
        phase1_output=phase1,
        run_id=rid,
        user_id=user_id,
    )
