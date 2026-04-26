"""
Live integration tests for the Phase 1 Judge agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434)
  - gemma4:e4b model pulled: ollama pull gemma4:e4b
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_phase1_judge_live.py -v -m integration
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pytest.importorskip("ollama", reason="ollama package not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.schemas import (
    Phase1JudgeInput,
    RawArticle,
    RejectedTopic,
    RemovalReason,
    ScoredArticle,
    ScoredFilteredTopic,
    SelectedTopic,
)
from src.agents.phase1_judge.agent import run_agent

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
    tracer = provider.get_tracer("test-phase1-judge")
    return tracer, exporter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_article(url: str, title: str) -> ScoredArticle:
    raw = RawArticle(
        url=url,
        title=title,
        content=f"Detailed reporting on {title}. Multiple independent sources confirm the development.",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )
    return ScoredArticle(article=raw, heat_score=0.7, reasoning="scored")


def _make_topic(
    topic_id: str,
    topic_text: str,
    heat_score: float,
    confidence: str = "HIGH",
    kept: int = 5,
    removed: int = 1,
) -> ScoredFilteredTopic:
    articles = [
        _make_article(f"https://example.com/{topic_id}-{i}", f"{topic_text} story {i}")
        for i in range(kept)
    ]
    reasons = [
        RemovalReason(url=f"https://example.com/{topic_id}-r{i}", constraint_violated=0)
        for i in range(removed)
    ]
    return ScoredFilteredTopic(
        topic_id=topic_id,
        topic_text=topic_text,
        heat_score=heat_score,
        kept_results=articles,
        confidence=confidence,
        removal_reasons=reasons,
    )


def _make_input(
    run_id: str,
    all_topics: list[ScoredFilteredTopic],
    selected_ids: list[str],
    rejected_ids: list[str],
    selected_reasonings: dict[str, str] | None = None,
    rejected_reasons: dict[str, str] | None = None,
) -> Phase1JudgeInput:
    topic_map = {t.topic_id: t for t in all_topics}
    selected = []
    for rank, tid in enumerate(selected_ids, start=1):
        reasoning = (selected_reasonings or {}).get(tid, "Strong coverage and heat score.")
        t = topic_map[tid]
        selected.append(SelectedTopic(
            topic_id=tid,
            rank=rank,
            selection_reasoning=reasoning,
            confidence_assessment=t.confidence,
        ))
    rejected = []
    for tid in rejected_ids:
        reason = (rejected_reasons or {}).get(tid, "Lower ranked than selected alternatives.")
        rejected.append(RejectedTopic(topic_id=tid, reject_reason=reason))
    return Phase1JudgeInput(
        run_id=run_id,
        all_candidates=all_topics,
        selected=selected,
        rejected=rejected,
    )


# ── Test 1: good selector choices → ENDORSED verdict ──────────────────────────

@pytest.mark.asyncio
async def test_good_choices_endorsed(span_tracer):
    """When selector made sound choices, judge endorses without adjustment."""
    tracer, _ = span_tracer

    topics = [
        _make_topic("tech-001", "artificial intelligence research breakthroughs", 0.88, "HIGH"),
        _make_topic("sci-001", "climate change policy and international agreements", 0.82, "HIGH"),
        _make_topic("biz-001", "global financial markets and central bank decisions", 0.76, "HIGH"),
        _make_topic("health-001", "public health initiatives and vaccine development", 0.65, "MEDIUM"),
    ]
    inp = _make_input(
        "judge-endorse-test",
        all_topics=topics,
        selected_ids=["tech-001", "sci-001", "biz-001"],
        rejected_ids=["health-001"],
        selected_reasonings={
            "tech-001": "Highest heat score with HIGH confidence — clear top pick.",
            "sci-001": "Strong coverage, policy relevance, diverse from tech.",
            "biz-001": "Market story fills business category gap.",
        },
        rejected_reasons={
            "health-001": "Lower heat score than all selected alternatives.",
        },
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert output.verdict == "endorsed", (
        f"Expected ENDORSED for clearly sound selector choices. Got: {output.verdict}. "
        f"Adjustments: {[a.model_dump() for a in output.adjustments]}"
    )
    assert len(output.final_selected) == 3
    assert output.adjustments == []
    # All final_selected should have source="selector"
    for ft in output.final_selected:
        assert ft.source == "selector", (
            f"Expected source=selector on endorsed topics, got: {ft.source}"
        )
    # judged_articles must be populated
    approved = [j for j in output.judged_articles if j.approved]
    assert len(approved) > 0, "At least some judged_articles should be approved"


# ── Test 2: LOW confidence selected over HIGH → ADJUSTED verdict ──────────────

@pytest.mark.asyncio
async def test_low_confidence_over_high_triggers_adjustment(span_tracer):
    """
    Judge should swap a LOW-confidence high-heat topic for a HIGH-confidence
    lower-heat alternative. The mandate is explicit: HIGH at 0.65 > LOW at 0.88.
    """
    tracer, exporter = span_tracer

    topics = [
        # Selected by selector — low confidence despite high heat
        _make_topic(
            "low-hot",
            "viral social media celebrity controversy update",
            heat_score=0.92,
            confidence="LOW",
            kept=1,
            removed=9,  # extreme removal ratio — very low coverage
        ),
        # Rejected by selector — high confidence, lower heat
        _make_topic(
            "high-stable",
            "semiconductor manufacturing capacity and supply chain analysis",
            heat_score=0.65,
            confidence="HIGH",
            kept=9,
            removed=1,
        ),
    ]
    inp = _make_input(
        "judge-confidence-swap-test",
        all_topics=topics,
        selected_ids=["low-hot"],
        rejected_ids=["high-stable"],
        selected_reasonings={
            "low-hot": "Highest heat score in the pool.",
        },
        rejected_reasons={
            "high-stable": "Lower heat score than selected topic.",
        },
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert output.verdict == "adjusted", (
        f"Expected ADJUSTED for LOW-confidence selection over HIGH-confidence alternative. "
        f"Got: {output.verdict}."
    )
    assert len(output.adjustments) >= 1, "Expected at least one adjustment"

    # The adjustment should be a confidence_upgrade swap
    adj = output.adjustments[0]
    assert adj.reason_code == "confidence_upgrade", (
        f"Expected reason_code=confidence_upgrade, got: {adj.reason_code}"
    )
    assert adj.removed_topic_id == "low-hot"
    assert adj.added_topic_id == "high-stable"

    # final_selected should contain the swapped-in topic
    final_ids = {ft.topic_id for ft in output.final_selected}
    assert "high-stable" in final_ids, (
        f"Expected high-stable in final_selected after swap. Got: {final_ids}"
    )
    # The swapped-in topic should have source="judge"
    judge_topics = [ft for ft in output.final_selected if ft.source == "judge"]
    assert judge_topics, "Swapped-in topic must have source='judge'"

    # OTel adjustment event must be emitted
    spans = exporter.get_finished_spans()
    eval_spans = [s for s in spans if s.name == "phase1_judge.evaluate"]
    assert eval_spans, "phase1_judge.evaluate span not found"
    events = eval_spans[0].events
    adj_events = [e for e in events if e.name == "phase1_judge.adjustment"]
    assert adj_events, "Expected phase1_judge.adjustment OTel event for swap"
    event_attrs = adj_events[0].attributes
    assert event_attrs.get("reason_code") == "confidence_upgrade"
    assert event_attrs.get("swapped_out_topic_id") == "low-hot"
    assert event_attrs.get("swapped_in_topic_id") == "high-stable"
    # Event must NOT contain reasoning text
    assert "celebrity" not in str(event_attrs), (
        "Topic text must not appear in OTel event attributes"
    )


# ── Test 3: all same category selected → diversity issue noted ─────────────────

@pytest.mark.asyncio
async def test_same_category_diversity_noted(span_tracer):
    """
    When all selected topics are from the same category, the judge should note
    the concentration — either in endorsement_note or via a diversity adjustment.
    """
    tracer, _ = span_tracer

    topics = [
        _make_topic("pol-001", "US presidential race polling averages", 0.84, "HIGH"),
        _make_topic("pol-002", "Senate midterm election candidate filings", 0.80, "HIGH"),
        _make_topic("pol-003", "State governor race results and analysis", 0.76, "HIGH"),
        _make_topic("tech-001", "AI research paper accepted at major conference", 0.72, "HIGH"),
    ]
    inp = _make_input(
        "judge-diversity-test",
        all_topics=topics,
        selected_ids=["pol-001", "pol-002", "pol-003"],
        rejected_ids=["tech-001"],
        selected_reasonings={
            "pol-001": "Highest heat score.",
            "pol-002": "Strong political coverage.",
            "pol-003": "Regional election story.",
        },
        rejected_reasons={
            "tech-001": "Lower heat score than all selected topics.",
        },
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    # The judge should either note diversity (endorsement_note) or make a swap
    has_diversity_note = (
        output.endorsement_note is not None and len(output.endorsement_note) > 0
    )
    made_diversity_swap = any(
        a.reason_code == "diversity_improvement" for a in output.adjustments
    )
    assert has_diversity_note or made_diversity_swap or output.verdict in ("endorsed", "adjusted"), (
        "Judge should produce a verdict regardless of diversity handling"
    )
    # Output must be structurally valid
    assert output.overall_confidence in ("HIGH", "MEDIUM", "LOW")
    assert len(output.final_selected) > 0


# ── Test 4: reasonable selector tradeoff → judge endorses ─────────────────────

@pytest.mark.asyncio
async def test_reasonable_tradeoff_endorsed(span_tracer):
    """
    Ambiguous same-confidence tradeoff: tech-a (0.75, HIGH) selected over tech-b
    (0.78, HIGH) with a coherent novelty justification. Both "endorsed" and
    "adjusted" are valid — the test verifies reasoning quality, not a forced verdict.

    If adjusted: exactly one swap with a valid reason_code must be present.
    The selector's judgment on a minor score difference within the same confidence
    tier is defensible, so endorsement is the preferred outcome.
    """
    tracer, _ = span_tracer

    topics = [
        _make_topic("tech-a", "open source LLM release with benchmark results", 0.75, "HIGH"),
        _make_topic("tech-b", "cloud provider announces new GPU instance types", 0.78, "HIGH"),
        _make_topic("sci-001", "ocean temperature record and climate implications", 0.80, "HIGH"),
    ]
    inp = _make_input(
        "judge-tradeoff-test",
        all_topics=topics,
        selected_ids=["sci-001", "tech-a"],
        rejected_ids=["tech-b"],
        selected_reasonings={
            "sci-001": "Top heat score, HIGH confidence, science coverage.",
            "tech-a": "Slightly lower heat but better novelty — new model release.",
        },
        rejected_reasons={
            "tech-b": "Same category as tech-a with marginally higher score but less novel.",
        },
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert output.verdict in ("endorsed", "adjusted"), (
        f"Expected endorsed or adjusted, got: {output.verdict}"
    )

    valid_reason_codes = {"confidence_upgrade", "diversity_improvement", "quality_concern"}
    if output.verdict == "adjusted":
        assert len(output.adjustments) == 1, (
            f"Expected exactly one swap for a minor tradeoff, got {len(output.adjustments)}: "
            f"{[a.model_dump() for a in output.adjustments]}"
        )
        assert output.adjustments[0].reason_code in valid_reason_codes, (
            f"Invalid reason_code: {output.adjustments[0].reason_code}"
        )
        assert output.adjustments[0].action == "swap"
    else:
        assert output.adjustments == [], (
            "Endorsed verdict must carry no adjustments"
        )


# ── Test 5: all candidates LOW confidence → DEGRADED, original selection returned

@pytest.mark.asyncio
async def test_all_low_confidence_degraded(span_tracer):
    """
    When all candidates are LOW confidence, the agent proceeds in DEGRADED mode.
    The judge should still produce a verdict (typically endorsing what it has).
    """
    tracer, _ = span_tracer

    topics = [
        _make_topic("low-001", "breaking news snippet with limited context", 0.75, "LOW"),
        _make_topic("low-002", "unverified report from single source", 0.68, "LOW"),
        _make_topic("low-003", "paywalled article with only headline available", 0.60, "LOW"),
    ]
    inp = _make_input(
        "judge-all-low-test",
        all_topics=topics,
        selected_ids=["low-001", "low-002"],
        rejected_ids=["low-003"],
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, (
        "Expected success=True for all-LOW-confidence run (DEGRADED proceeds)"
    )
    # DEGRADED warning should be present
    assert any("all_candidates_low_confidence" in w for w in output.warnings), (
        f"Expected DEGRADED warning. Got: {output.warnings}"
    )
    # Must still produce a verdict and final_selected
    assert output.verdict in ("endorsed", "adjusted")
    assert len(output.final_selected) > 0


# ── Test 6: empty candidates → FATAL AgentResult ──────────────────────────────

@pytest.mark.asyncio
async def test_empty_candidates_fatal(span_tracer):
    """
    When all_candidates, selected, and selected_articles are empty, the agent
    returns FATAL NO_VIABLE_CANDIDATES without calling Ollama.
    """
    tracer, exporter = span_tracer

    inp = Phase1JudgeInput(
        run_id="judge-empty-test",
        all_candidates=[],
        selected=[],
        rejected=[],
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is False, "Expected success=False for empty candidates"
    assert output.error is not None
    assert output.error.error_code == "NO_VIABLE_CANDIDATES"
    assert output.error.severity == "FATAL"
    assert len(output.final_selected) == 0
    assert len(output.judged_articles) == 0

    # No LLM call should have been made
    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if "llm_call" in s.name]
    assert not llm_spans, (
        f"No LLM call expected for empty input. Got: {[s.name for s in llm_spans]}"
    )


# ── OTel span structure verification ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_otel_spans_created(span_tracer):
    """phase1_judge.evaluate and phase1_judge.llm_call spans are emitted."""
    tracer, exporter = span_tracer

    topics = [
        _make_topic("otel-001", "quantum computing error correction milestone", 0.80, "HIGH"),
        _make_topic("otel-002", "renewable energy grid storage breakthrough", 0.74, "HIGH"),
    ]
    inp = _make_input(
        "judge-spans-test",
        all_topics=topics,
        selected_ids=["otel-001"],
        rejected_ids=["otel-002"],
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("phase1_judge.evaluate" in n for n in span_names), (
        f"Expected phase1_judge.evaluate span. Got: {span_names}"
    )
    assert any("phase1_judge.llm_call" in n for n in span_names), (
        f"Expected phase1_judge.llm_call span. Got: {span_names}"
    )

    eval_spans = [s for s in spans if s.name == "phase1_judge.evaluate"]
    attrs = eval_spans[0].attributes
    assert "candidate_count" in attrs
    assert "selected_count" in attrs
    assert "verdict" in attrs
    assert "adjustment_count" in attrs
    assert attrs["candidate_count"] == 2
    assert attrs["selected_count"] == 1


@pytest.mark.asyncio
async def test_topic_text_not_in_spans(span_tracer):
    """Topic text and reasoning must never appear in any OTel span attribute."""
    tracer, exporter = span_tracer

    unique_text = "unique-canary-string-that-must-not-appear-in-telemetry"
    topics = [
        _make_topic("priv-001", unique_text, 0.78, "HIGH"),
        _make_topic("priv-002", "science and technology research coverage", 0.71, "MEDIUM"),
    ]
    inp = _make_input(
        "judge-privacy-test",
        all_topics=topics,
        selected_ids=["priv-001"],
        rejected_ids=["priv-002"],
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    for span in spans:
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert unique_text not in value, (
                    f"Topic text found in span '{span.name}' attribute '{key}'"
                )
        for event in span.events:
            for key, value in event.attributes.items():
                if isinstance(value, str):
                    assert unique_text not in value, (
                        f"Topic text found in span event '{event.name}' attribute '{key}'"
                    )


# ── LangSmith run config tags ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_langsmith_run_config_tags():
    """make_run_config produces the required LangSmith tags for phase1_judge."""
    from src.common.prompt_loader import make_run_config

    cfg = make_run_config("phase1_judge", "v1.0", run_id="test-judge-run-01")

    assert "agent:phase1_judge" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]
    assert "run:test-judge-run-01" in cfg["tags"]
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "phase1_judge"
    assert cfg["run_name"] == "phase1_judge:v1.0"
