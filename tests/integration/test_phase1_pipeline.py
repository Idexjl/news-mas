"""
Integration test for the full Phase 1 pipeline.

Requires:
  - INTEGRATION_TESTS=true
  - Ollama running with gemma4:e4b pulled
  - TAVILY_API_KEY set
  - Optionally LANGSMITH_API_KEY for trace capture

Run:
  INTEGRATION_TESTS=true pytest tests/integration/test_phase1_pipeline.py -v -m "integration and slow"
"""
import os
import uuid

import pytest

from src.common.schemas import Phase1Output, RankedCandidate, TopicInput
from src.orchestrators.phase1.runner import run_phase1

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _skip_if_not_enabled():
    if os.getenv("INTEGRATION_TESTS", "false").lower() not in ("true", "1", "yes"):
        pytest.skip("Set INTEGRATION_TESTS=true to run this test")


TOPICS = [
    TopicInput(topic_id="t-ai", topic_text="artificial intelligence news"),
    TopicInput(topic_id="t-rockies", topic_text="Colorado Rockies baseball"),
    TopicInput(topic_id="t-python", topic_text="Python programming language"),
]


@pytest.fixture()
def run_id() -> str:
    return f"integ-phase1-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_phase1_pipeline_completes(run_id: str):
    """Pipeline runs to completion and returns a well-formed Phase1Output."""
    _skip_if_not_enabled()

    result = await run_phase1(
        run_id=run_id,
        topics=TOPICS,
        since_date="2025-01-01",
        user_id="test-user",
    )

    assert isinstance(result, Phase1Output)
    assert result.run_id == run_id
    assert result.run_status in ("complete", "no_candidates", "partial"), (
        f"Unexpected run_status: {result.run_status!r}"
    )
    assert isinstance(result.ranked_candidates, list)
    assert isinstance(result.errors, list)


@pytest.mark.asyncio
async def test_phase1_at_least_one_non_tombstone(run_id: str):
    """At least one topic should survive the full pipeline."""
    _skip_if_not_enabled()

    result = await run_phase1(
        run_id=run_id,
        topics=TOPICS,
        since_date="2025-01-01",
        user_id="test-user",
    )

    non_tombstones = [c for c in result.ranked_candidates if not c.is_tombstone]
    assert non_tombstones, (
        f"All topics were tombstoned. run_status={result.run_status!r}, "
        f"errors={[e.error_code for e in result.errors]}"
    )


@pytest.mark.asyncio
async def test_phase1_ranked_candidates_have_required_fields(run_id: str):
    """Non-tombstone candidates must carry topic_id, rank, source, and confidence."""
    _skip_if_not_enabled()

    result = await run_phase1(
        run_id=run_id,
        topics=TOPICS,
        since_date="2025-01-01",
        user_id="test-user",
    )

    for cand in result.ranked_candidates:
        assert isinstance(cand, RankedCandidate)
        assert cand.topic_id
        assert cand.rank >= 1
        if not cand.is_tombstone:
            assert cand.source in ("selector", "judge")
            assert cand.confidence in ("HIGH", "MEDIUM", "LOW")


@pytest.mark.asyncio
async def test_phase1_token_budget_not_exceeded(run_id: str):
    """Token usage should stay well within a generous default ceiling."""
    _skip_if_not_enabled()

    result = await run_phase1(
        run_id=run_id,
        topics=TOPICS,
        since_date="2025-01-01",
        user_id="test-user",
    )

    assert result.run_status != "budget_exceeded", (
        f"Token budget was exceeded: {result.total_tokens_used} tokens used"
    )
    # Sanity check: three topics with moderate article counts should stay under 500k
    assert result.total_tokens_used < 500_000, (
        f"Token usage suspiciously high: {result.total_tokens_used}"
    )


@pytest.mark.asyncio
async def test_phase1_circuit_breaker_blocks_concurrent_run():
    """A second run with the same run_id is blocked if the first is still 'running'."""
    _skip_if_not_enabled()

    fernet_key = os.getenv("FERNET_KEY")
    if not fernet_key:
        pytest.skip("FERNET_KEY not set — circuit breaker requires storage")

    from src.common.storage import FernetStorage, RunRepository
    from datetime import datetime, timezone

    run_id_a = f"integ-cb-{uuid.uuid4().hex[:8]}"

    repo = RunRepository(FernetStorage())
    repo.save(run_id_a, {
        "run_id": run_id_a,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "topics": [],
    })

    try:
        new_run_id = f"integ-cb-blocked-{uuid.uuid4().hex[:8]}"
        result = await run_phase1(
            run_id=new_run_id,
            topics=TOPICS[:1],
            since_date="2025-01-01",
            user_id="test-user",
        )
        assert result.run_status == "circuit_breaker_active", (
            f"Expected circuit_breaker_active, got {result.run_status!r}"
        )
    finally:
        # Clean up: mark the fake run as complete so it doesn't block further tests
        repo.save(run_id_a, {
            "run_id": run_id_a,
            "status": "complete",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "topics": [],
        })


@pytest.mark.asyncio
async def test_phase1_empty_topics_returns_no_topics():
    """Calling run_phase1 with no topics should return run_status='no_topics'."""
    _skip_if_not_enabled()

    run_id = f"integ-empty-{uuid.uuid4().hex[:8]}"
    result = await run_phase1(
        run_id=run_id,
        topics=[],
        since_date="2025-01-01",
        user_id="test-user",
    )
    assert result.run_status == "no_topics"
    assert result.ranked_candidates == []
