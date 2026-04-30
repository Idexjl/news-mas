"""
Live integration tests for the Relevance Gate agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - ANTHROPIC_API_KEY set (default model: claude-haiku-4-5-20251001)
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_relevance_gate_live.py -v -m integration
"""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("anthropic", reason="anthropic package not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.schemas import RelevanceGateInput, RelevanceGateOutput
from src.agents.relevance_gate.agent import run_agent

_INTEGRATION = os.getenv("INTEGRATION_TESTS", "").lower() == "true"

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_integration():
    if not _INTEGRATION:
        pytest.skip("Set INTEGRATION_TESTS=true to run live integration tests")


@pytest.fixture()
def span_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-relevance-gate")
    return tracer, exporter


# ── Input builders ─────────────────────────────────────────────────────────────

def _make_inp(
    *,
    run_id: str = "gate-test-00",
    topic_id: str = "topic-abc",
    topic_text: str = "quantum computing breakthroughs",
    heat_score: float = 0.5,
    overall_confidence: str = "MEDIUM",
    reviewer_verdict: str = "pass",
    retry_count: int = 0,
    budget_exhausted: bool = False,
    reviewer_last_feedback: str | None = None,
    tombstone_reason: str | None = None,
) -> RelevanceGateInput:
    return RelevanceGateInput(
        run_id=run_id,
        topic_id=topic_id,
        topic_text=topic_text,
        heat_score=heat_score,
        overall_confidence=overall_confidence,  # type: ignore[arg-type]
        reviewer_verdict=reviewer_verdict,       # type: ignore[arg-type]
        retry_count=retry_count,
        budget_exhausted=budget_exhausted,
        reviewer_last_feedback=reviewer_last_feedback,
        tombstone_reason=tombstone_reason,
    )


# ── Test 1: high heat + HIGH confidence → fast PASS ───────────────────────────

@pytest.mark.asyncio
async def test_high_heat_high_confidence_fast_pass(span_tracer):
    """
    heat >= 0.8 AND confidence == HIGH is always a fast PASS.
    No LLM call should be made.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-01",
        heat_score=0.9,
        overall_confidence="HIGH",
        reviewer_verdict="pass",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
    ) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.success is True
    assert output.decision == "pass", f"Expected pass, got: {output.decision}"
    assert output.gate_path == "fast_pass"
    assert output.tombstone is None

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "relevance_gate.evaluate" in span_names
    assert "relevance_gate.llm_call" not in span_names, "LLM span must not exist on fast path"

    eval_span = next(s for s in spans if s.name == "relevance_gate.evaluate")
    assert eval_span.attributes.get("gate_path") == "fast_pass"
    assert eval_span.attributes.get("decision") == "pass"


# ── Test 2: budget_exhausted → fast TOMBSTONE ─────────────────────────────────

@pytest.mark.asyncio
async def test_budget_exhausted_fast_tombstone(span_tracer):
    """
    budget_exhausted=True is always a fast TOMBSTONE, no LLM call.
    reason_code must be 'budget_exhausted'.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-02",
        heat_score=0.6,
        overall_confidence="HIGH",
        budget_exhausted=True,
        tombstone_reason="Summarizer retries exhausted after 3 attempts.",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
    ) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.success is True
    assert output.decision == "tombstone"
    assert output.gate_path == "fast_tombstone"
    assert output.tombstone is not None
    assert output.tombstone.reason_code == "budget_exhausted"

    # OTel: tombstone event on evaluate span
    spans = exporter.get_finished_spans()
    eval_span = next(s for s in spans if s.name == "relevance_gate.evaluate")
    tombstone_events = [e for e in eval_span.events if e.name == "relevance_gate.tombstone"]
    assert tombstone_events, "Expected relevance_gate.tombstone OTel event"
    assert tombstone_events[0].attributes.get("reason_code") == "budget_exhausted"

    # LLM call span must be absent
    assert "relevance_gate.llm_call" not in [s.name for s in spans]


# ── Test 3: very cold topic → fast TOMBSTONE ──────────────────────────────────

@pytest.mark.asyncio
async def test_cold_topic_fast_tombstone(span_tracer):
    """
    heat_score < 0.2 is always a fast TOMBSTONE regardless of confidence.
    reason_code must be 'too_cold'.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-03",
        heat_score=0.1,
        overall_confidence="HIGH",
        reviewer_verdict="pass",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
    ) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.decision == "tombstone"
    assert output.gate_path == "fast_tombstone"
    assert output.tombstone is not None
    assert output.tombstone.reason_code == "too_cold"

    spans = exporter.get_finished_spans()
    assert "relevance_gate.llm_call" not in [s.name for s in spans]

    eval_span = next(s for s in spans if s.name == "relevance_gate.evaluate")
    assert eval_span.attributes.get("gate_path") == "fast_tombstone"


# ── Test 4: medium heat + MEDIUM confidence → LLM path ───────────────────────

@pytest.mark.asyncio
async def test_medium_heat_medium_confidence_llm_path(span_tracer):
    """
    Borderline input (heat=0.5, MEDIUM confidence, reviewer passed) must go
    through the LLM path and return a valid decision.
    If tombstoned, a tombstone_message must be present.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-04",
        heat_score=0.5,
        overall_confidence="MEDIUM",
        reviewer_verdict="pass",
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert output.gate_path == "llm_decision"
    assert output.decision in ("pass", "tombstone"), f"Unexpected decision: {output.decision}"

    if output.decision == "tombstone":
        assert output.tombstone is not None
        assert output.tombstone.tombstone_message, "Tombstoned topic must have a message"
        assert output.tombstone.reason_code == "gate_decision"

    # OTel: llm_call span must be present
    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "relevance_gate.llm_call" in span_names, (
        f"Expected llm_call span on LLM path. Got: {span_names}"
    )
    llm_span = next(s for s in spans if s.name == "relevance_gate.llm_call")
    assert llm_span.attributes.get("prompt_version") == "v1.0"


# ── Test 5: LOW confidence + low heat → fast TOMBSTONE ───────────────────────

@pytest.mark.asyncio
async def test_low_confidence_low_heat_fast_tombstone(span_tracer):
    """
    overall_confidence == LOW AND heat_score < 0.4 → fast TOMBSTONE.
    reason_code must be 'low_confidence'.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-05",
        heat_score=0.3,
        overall_confidence="LOW",
        reviewer_verdict="pass",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
    ) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.decision == "tombstone"
    assert output.gate_path == "fast_tombstone"
    assert output.tombstone is not None
    assert output.tombstone.reason_code == "low_confidence"

    spans = exporter.get_finished_spans()
    assert "relevance_gate.llm_call" not in [s.name for s in spans]


# ── Test 6: reviewer passed + good heat → fast PASS ──────────────────────────

@pytest.mark.asyncio
async def test_reviewer_pass_good_heat_fast_pass(span_tracer):
    """
    reviewer_verdict == 'pass' AND heat_score >= 0.7 → fast PASS, no LLM.
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-06",
        heat_score=0.75,
        overall_confidence="MEDIUM",
        reviewer_verdict="pass",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
    ) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.success is True
    assert output.decision == "pass"
    assert output.gate_path == "fast_pass"
    assert output.tombstone is None

    spans = exporter.get_finished_spans()
    assert "relevance_gate.llm_call" not in [s.name for s in spans]
    eval_span = next(s for s in spans if s.name == "relevance_gate.evaluate")
    assert eval_span.attributes.get("gate_path") == "fast_pass"


# ── Test 7: LLM unavailable → DEGRADED PASS ──────────────────────────────────

@pytest.mark.asyncio
async def test_llm_unavailable_degraded_pass(span_tracer):
    """
    When the LLM raises, the gate must fall back to DEGRADED pass.
    - decision == 'pass' (safe default)
    - success == True (pipeline continues)
    - error.severity == DEGRADED
    - 'gate_llm_unavailable' in warnings
    """
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-test-07",
        heat_score=0.5,
        overall_confidence="MEDIUM",
        reviewer_verdict="pass",
    )

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Connection refused"),
    ):
        output = await run_agent(inp, tracer=tracer)

    assert output.success is True, "Expected success=True for DEGRADED fallback"
    assert output.decision == "pass", f"Expected pass on LLM failure, got: {output.decision}"
    assert output.gate_path == "llm_decision"
    assert "gate_llm_unavailable" in output.warnings, (
        f"Expected warning. Got: {output.warnings}"
    )
    assert output.error is not None
    assert output.error.severity == "DEGRADED", (
        f"Expected DEGRADED severity, got: {output.error.severity}"
    )
    assert output.tombstone is None


# ── Test 8: tombstone message is user-friendly ────────────────────────────────

@pytest.mark.asyncio
async def test_tombstone_message_user_friendly(span_tracer):
    """
    When the LLM decides to tombstone, the tombstone_message must:
    - be a non-empty string
    - not contain the raw topic_text
    - be readable prose (not JSON fragments or technical terms)
    """
    tracer, exporter = span_tracer
    topic_text = "unique-canary-topic-x7q9z"
    inp = _make_inp(
        run_id="gate-test-08",
        topic_id="topic-canary",
        topic_text=topic_text,
        heat_score=0.35,
        overall_confidence="MEDIUM",
        reviewer_verdict="pass",
    )

    mock_response = json.dumps({
        "decision": "tombstone",
        "reasoning": "Heat is moderate and confidence does not justify inclusion.",
        "tombstone_message": "Limited significant coverage for this topic this week.",
    })

    with patch(
        "src.agents.relevance_gate.agent._call_anthropic",
        new_callable=AsyncMock,
        return_value={"message": {"content": mock_response}},
    ):
        output = await run_agent(inp, tracer=tracer)

    assert output.decision == "tombstone"
    assert output.tombstone is not None
    msg = output.tombstone.tombstone_message
    assert msg, "Tombstone message must be non-empty"
    assert topic_text not in msg, (
        f"Tombstone message must not contain topic_text. Got: {msg!r}"
    )
    assert len(msg.split()) >= 3, "Tombstone message must be readable prose"

    # tombstone_message must not appear in any OTel span attribute or event
    spans = exporter.get_finished_spans()
    for span in spans:
        for value in span.attributes.values():
            if isinstance(value, str):
                assert msg not in value, (
                    f"tombstone_message leaked into span '{span.name}' attribute"
                )
        for event in span.events:
            for value in event.attributes.values():
                if isinstance(value, str):
                    assert msg not in value, (
                        f"tombstone_message leaked into span '{span.name}' event '{event.name}'"
                    )


# ── OTel span structure validation ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_otel_evaluate_span_attributes_fast_pass(span_tracer):
    """All required OTel attributes are present on the evaluate span for fast pass."""
    tracer, exporter = span_tracer
    inp = _make_inp(run_id="gate-otel-01", heat_score=0.85, overall_confidence="HIGH")

    with patch("src.agents.relevance_gate.agent._call_anthropic", new_callable=AsyncMock):
        output = await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    eval_span = next((s for s in spans if s.name == "relevance_gate.evaluate"), None)
    assert eval_span is not None, "Expected relevance_gate.evaluate span"

    required_attrs = {"topic_id", "run_id", "gate_path", "heat_score", "confidence",
                      "retry_count", "budget_exhausted", "decision"}
    for attr in required_attrs:
        assert attr in eval_span.attributes, f"Missing attribute '{attr}' on evaluate span"

    # topic_text must never appear in any span attribute
    for span in spans:
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert inp.topic_text not in value, (
                    f"topic_text leaked into span '{span.name}' attribute '{key}'"
                )


@pytest.mark.asyncio
async def test_otel_tombstone_event_never_contains_message(span_tracer):
    """The tombstone OTel event must contain reason_code but never tombstone_message."""
    tracer, exporter = span_tracer
    inp = _make_inp(
        run_id="gate-otel-02",
        heat_score=0.1,   # triggers fast tombstone: too_cold
        overall_confidence="HIGH",
    )

    with patch("src.agents.relevance_gate.agent._call_anthropic", new_callable=AsyncMock):
        output = await run_agent(inp, tracer=tracer)

    assert output.decision == "tombstone"
    assert output.tombstone is not None
    tombstone_msg = output.tombstone.tombstone_message

    spans = exporter.get_finished_spans()
    eval_span = next(s for s in spans if s.name == "relevance_gate.evaluate")
    tombstone_events = [e for e in eval_span.events if e.name == "relevance_gate.tombstone"]
    assert tombstone_events, "Expected relevance_gate.tombstone event"

    event_attrs = tombstone_events[0].attributes
    assert "reason_code" in event_attrs
    # tombstone_message must not be in the event
    for key, value in event_attrs.items():
        assert key != "tombstone_message", "tombstone_message must never appear in OTel events"
        if isinstance(value, str):
            assert tombstone_msg not in value, (
                f"tombstone_message content leaked into event attribute '{key}'"
            )


@pytest.mark.asyncio
async def test_no_llm_call_span_on_fast_paths(span_tracer):
    """Fast paths must not produce a relevance_gate.llm_call span."""
    tracer, exporter = span_tracer

    fast_inputs = [
        # fast_pass
        _make_inp(run_id="gate-nollm-01", heat_score=0.9, overall_confidence="HIGH"),
        # fast_tombstone: too_cold
        _make_inp(run_id="gate-nollm-02", heat_score=0.05),
        # fast_tombstone: budget_exhausted
        _make_inp(run_id="gate-nollm-03", budget_exhausted=True),
        # fast_tombstone: low_confidence
        _make_inp(run_id="gate-nollm-04", heat_score=0.3, overall_confidence="LOW"),
        # fast_pass: reviewer pass + good heat
        _make_inp(run_id="gate-nollm-05", heat_score=0.75, reviewer_verdict="pass"),
    ]

    for inp in fast_inputs:
        exporter.clear()
        with patch("src.agents.relevance_gate.agent._call_anthropic", new_callable=AsyncMock) as m:
            await run_agent(inp, tracer=tracer)
        m.assert_not_called()
        spans = exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name == "relevance_gate.llm_call"]
        assert not llm_spans, (
            f"Got unexpected llm_call span for run_id={inp.run_id}"
        )


# ── LangSmith run config ──────────────────────────────────────────────────────

def test_langsmith_run_config_tags():
    """make_run_config produces required LangSmith tags for the relevance gate."""
    from src.common.prompt_loader import make_run_config

    cfg = make_run_config("relevance_gate", "v1.0", run_id="gate-cfg-01", topic_id="t1")

    assert "agent:relevance_gate" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]
    assert "run:gate-cfg-01" in cfg["tags"]
    assert "topic:t1" in cfg["tags"]
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "relevance_gate"


# ── Data quality: out-of-range inputs ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_heat_score_out_of_range_defaults_to_zero(span_tracer):
    """heat_score outside [0,1] is normalised to 0.0 → fast tombstone (too_cold)."""
    tracer, exporter = span_tracer
    inp = _make_inp(run_id="gate-dq-01", heat_score=5.0)

    with patch("src.agents.relevance_gate.agent._call_anthropic", new_callable=AsyncMock) as m:
        output = await run_agent(inp, tracer=tracer)

    m.assert_not_called()
    # heat normalised to 0.0, which is < 0.2 → fast_tombstone / too_cold
    assert output.decision == "tombstone"
    assert output.tombstone.reason_code == "too_cold"


@pytest.mark.asyncio
async def test_invalid_confidence_defaults_to_low(span_tracer):
    """
    Invalid overall_confidence is normalised to LOW → conservative behaviour.

    Uses model_construct() to bypass Pydantic validation, simulating bad data
    arriving from an internal caller (e.g. a buggy orchestrator) rather than
    an HTTP endpoint where Pydantic would have already rejected it.
    """
    tracer, exporter = span_tracer
    # model_construct() skips all validators — lets us set an invalid Literal value
    inp = RelevanceGateInput.model_construct(
        run_id="gate-dq-02",
        topic_id="topic-abc",
        topic_text="quantum computing breakthroughs",
        heat_score=0.35,
        overall_confidence="UNKNOWN",  # invalid — normaliser must coerce to LOW
        reviewer_verdict="pass",
        retry_count=0,
        budget_exhausted=False,
        reviewer_last_feedback=None,
        tombstone_reason=None,
        aap_token=None,
    )

    # With confidence normalised to LOW and heat=0.35 < 0.4 → fast_tombstone
    with patch("src.agents.relevance_gate.agent._call_anthropic", new_callable=AsyncMock) as m:
        output = await run_agent(inp, tracer=tracer)

    m.assert_not_called()
    assert output.decision == "tombstone"
    assert output.tombstone.reason_code == "low_confidence"
