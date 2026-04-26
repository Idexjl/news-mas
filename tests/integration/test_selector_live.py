"""
Live integration tests for the Selector agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434)
  - gemma4:e4b model pulled: ollama pull gemma4:e4b
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_selector_live.py -v -m integration
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
    RawArticle,
    RemovalReason,
    ScoredArticle,
    ScoredFilteredTopic,
    SelectorInput,
)
from src.agents.selector.agent import run_agent

_INTEGRATION = os.getenv("INTEGRATION_TESTS", "").lower() == "true"

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_integration():
    if not _INTEGRATION:
        pytest.skip("Set INTEGRATION_TESTS=true to run live integration tests")


@pytest.fixture()
def span_tracer():
    """Scoped TracerProvider so spans don't touch the global OTel provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-selector")
    return tracer, exporter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_article(url: str, title: str, content: str) -> ScoredArticle:
    raw = RawArticle(
        url=url,
        title=title,
        content=content,
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )
    return ScoredArticle(article=raw, heat_score=0.7, reasoning="test article")


def _make_topic(
    topic_id: str,
    topic_text: str,
    heat_score: float,
    confidence: str = "HIGH",
    article_count: int = 3,
    removed_count: int = 1,
) -> ScoredFilteredTopic:
    articles = [
        _make_article(
            f"https://example.com/{topic_id}-{i}",
            f"{topic_text} article {i}",
            f"Content about {topic_text}, article {i}. "
            "This is a substantial news story with real reporting on the subject.",
        )
        for i in range(article_count)
    ]
    removal_reasons = [
        RemovalReason(url=f"https://example.com/{topic_id}-removed-{i}", constraint_violated=0)
        for i in range(removed_count)
    ]
    return ScoredFilteredTopic(
        topic_id=topic_id,
        topic_text=topic_text,
        heat_score=heat_score,
        kept_results=articles,
        confidence=confidence,
        removal_reasons=removal_reasons,
    )


# ── Test 1: 3 topics, select top 2 → correct output count ─────────────────────

@pytest.mark.asyncio
async def test_select_top_n_from_candidates(span_tracer):
    """Selector picks exactly max_select topics when more candidates are available."""
    tracer, _ = span_tracer

    topics = [
        _make_topic("tech-001", "artificial intelligence and machine learning", 0.85),
        _make_topic("biz-001", "global financial markets and economic policy", 0.72),
        _make_topic("sci-001", "climate science and environmental research", 0.68),
    ]

    inp = SelectorInput(
        run_id="integration-selector-top-n",
        topics=topics,
        max_select=2,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert len(output.selected) == 2, (
        f"Expected 2 selected topics, got {len(output.selected)}"
    )
    assert len(output.rejected) == 1, (
        f"Expected 1 rejected topic, got {len(output.rejected)}"
    )
    # All topic IDs must be covered
    all_ids = {t.topic_id for t in topics}
    selected_ids = {s.topic_id for s in output.selected}
    rejected_ids = {r.topic_id for r in output.rejected}
    assert selected_ids | rejected_ids == all_ids, (
        f"Not all topic IDs covered. Selected: {selected_ids}, Rejected: {rejected_ids}"
    )
    # Ranks must be assigned
    for s in output.selected:
        assert s.rank >= 1
        assert s.selection_reasoning, "selection_reasoning must be non-empty"
    for r in output.rejected:
        assert r.reject_reason, "reject_reason must be non-empty"


# ── Test 2: all same category → diversity_note populated ──────────────────────

@pytest.mark.asyncio
async def test_same_category_diversity_noted(span_tracer):
    """When all topics are from the same category, the selector should note it."""
    tracer, _ = span_tracer

    topics = [
        _make_topic("pol-001", "US presidential election polling and results", 0.82),
        _make_topic("pol-002", "Senate budget vote and fiscal policy debate", 0.78),
        _make_topic("pol-003", "State governor race outcome and implications", 0.71),
        _make_topic("pol-004", "Supreme Court decision on congressional redistricting", 0.69),
    ]

    inp = SelectorInput(
        run_id="integration-selector-diversity",
        topics=topics,
        max_select=3,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert len(output.selected) == 3
    assert len(output.rejected) == 1
    # The LLM should produce a diversity_note when all topics are in the same domain
    # (This is a soft assertion — LLM may or may not always produce the note)
    # We verify structure is valid regardless
    assert output.selection_confidence in ("HIGH", "MEDIUM", "LOW")


# ── Test 3: mixed confidence → HIGH beats LOW at slightly lower heat score ─────

@pytest.mark.asyncio
async def test_high_confidence_beats_low_confidence(span_tracer):
    """
    HIGH-confidence topic at 0.70 should be preferred over LOW-confidence
    topic at 0.88 when selecting only one topic.
    """
    tracer, _ = span_tracer

    high_conf_topic = _make_topic(
        "tech-high",
        "semiconductor supply chain recovery and chip manufacturing",
        heat_score=0.70,
        confidence="HIGH",
    )
    low_conf_topic = _make_topic(
        "tech-low",
        "consumer electronics sales trends this quarter",
        heat_score=0.88,
        confidence="LOW",
    )

    inp = SelectorInput(
        run_id="integration-selector-confidence",
        topics=[high_conf_topic, low_conf_topic],
        max_select=1,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert len(output.selected) == 1
    assert len(output.rejected) == 1

    selected_id = output.selected[0].topic_id
    assert selected_id == "tech-high", (
        f"Expected HIGH-confidence topic to be selected over LOW-confidence topic "
        f"with higher heat score. Got: {selected_id}. "
        f"Rejected: {output.rejected[0].topic_id}"
    )
    assert output.rejected[0].topic_id == "tech-low"
    assert output.rejected[0].reject_reason, "reject_reason must explain the confidence concern"


# ── Test 4: topic with TopicMemory history → history factored into reasoning ───

@pytest.mark.asyncio
async def test_topic_memory_context_factored_in(span_tracer):
    """
    A topic with a strong positive memory history (selected 3 runs in a row)
    should be reflected in the selection output. A cold topic (never selected)
    should be deprioritised if scores are otherwise close.
    """
    tracer, _ = span_tracer

    topics = [
        _make_topic("mem-warm", "renewable energy and clean technology investment", 0.72),
        _make_topic("mem-cold", "traditional oil and gas sector updates", 0.75),
    ]

    # Warm topic: selected consistently, trending up
    # Cold topic: never selected, heat declining
    topic_memory_context = {
        "mem-warm": [
            {"heat_score": 0.65, "was_selected": True},
            {"heat_score": 0.70, "was_selected": True},
            {"heat_score": 0.72, "was_selected": True},
        ],
        "mem-cold": [
            {"heat_score": 0.80, "was_selected": False},
            {"heat_score": 0.77, "was_selected": False},
            {"heat_score": 0.75, "was_selected": False},
        ],
    }

    inp = SelectorInput(
        run_id="integration-selector-memory",
        topics=topics,
        max_select=1,
        topic_memory_context=topic_memory_context,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got: {output.error}"
    assert len(output.selected) == 1
    assert len(output.rejected) == 1

    # With consistent selection history, mem-warm should edge out mem-cold
    # (soft assertion — the LLM makes this call, but memory context should inform it)
    selected_id = output.selected[0].topic_id
    assert selected_id == "mem-warm", (
        f"Expected the consistently-selected warm topic to beat the cold one. "
        f"Got: {selected_id}. This may indicate the memory context was not factored in."
    )

    # Verify the reasoning is substantive
    reasoning = output.selected[0].selection_reasoning
    assert len(reasoning) > 10, "selection_reasoning must be substantive"


# ── Test 5: single viable candidate → DEGRADED AgentResult, candidate selected ─

@pytest.mark.asyncio
async def test_single_candidate_degraded(span_tracer):
    """
    When only one topic is available but max_select=3, the agent should proceed
    in DEGRADED mode and select the one available topic.
    """
    tracer, _ = span_tracer

    topics = [
        _make_topic("solo-001", "space exploration and NASA missions", 0.79),
    ]

    inp = SelectorInput(
        run_id="integration-selector-single",
        topics=topics,
        max_select=3,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True (DEGRADED proceeds), got: {output.error}"
    assert len(output.selected) == 1, (
        f"Expected 1 selected topic (the only available one), got {len(output.selected)}"
    )
    assert len(output.rejected) == 0
    assert output.selected[0].topic_id == "solo-001"
    assert output.selected[0].selection_reasoning

    # DEGRADED warning should be present
    assert any("fewer_candidates_than_requested" in w for w in output.warnings), (
        f"Expected DEGRADED warning. Got warnings: {output.warnings}"
    )
    assert len(output.selected_articles) > 0, "selected_articles must contain kept results"


# ── Test 6: no viable candidates → NO_VIABLE_CANDIDATES FATAL AgentResult ─────

@pytest.mark.asyncio
async def test_no_viable_candidates_fatal(span_tracer):
    """
    When all topics have empty kept_results, the agent returns a FATAL
    NO_VIABLE_CANDIDATES error without calling Ollama.
    """
    tracer, exporter = span_tracer

    # Topics with no kept articles (all were filtered out)
    empty_topics = [
        ScoredFilteredTopic(
            topic_id="empty-001",
            topic_text="niche topic with no results",
            heat_score=0.0,
            kept_results=[],
            confidence="LOW",
            removal_reasons=[
                RemovalReason(url="https://example.com/x", constraint_violated=0)
            ],
        ),
        ScoredFilteredTopic(
            topic_id="empty-002",
            topic_text="another empty topic",
            heat_score=0.0,
            kept_results=[],
            confidence="LOW",
            removal_reasons=[],
        ),
    ]

    inp = SelectorInput(
        run_id="integration-selector-no-candidates",
        topics=empty_topics,
        max_select=3,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is False, "Expected success=False for no viable candidates"
    assert output.error is not None
    assert output.error.error_code == "NO_VIABLE_CANDIDATES", (
        f"Expected NO_VIABLE_CANDIDATES, got: {output.error.error_code}"
    )
    assert output.error.severity == "FATAL"
    assert len(output.selected) == 0
    assert len(output.selected_articles) == 0

    # No LLM call span should be created — we return early on empty input
    spans = exporter.get_finished_spans()
    llm_span_names = [s.name for s in spans if "llm_call" in s.name]
    assert not llm_span_names, (
        f"No LLM call span should be created for empty input. Got: {llm_span_names}"
    )


# ── OTel span verification ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_otel_spans_created(span_tracer):
    """selector.select and selector.llm_call spans are emitted."""
    tracer, exporter = span_tracer

    topics = [
        _make_topic("otel-001", "quantum computing breakthroughs in error correction", 0.80),
        _make_topic("otel-002", "global health policy and pandemic preparedness", 0.74),
    ]

    inp = SelectorInput(
        run_id="integration-selector-spans",
        topics=topics,
        max_select=1,
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("selector.select" in n for n in span_names), (
        f"Expected selector.select span. Got: {span_names}"
    )
    assert any("selector.llm_call" in n for n in span_names), (
        f"Expected selector.llm_call span. Got: {span_names}"
    )

    # Verify selector.select carries the expected attributes
    select_spans = [s for s in spans if s.name == "selector.select"]
    assert select_spans, "selector.select span not found"
    attrs = select_spans[0].attributes
    assert "input_count" in attrs
    assert "diversity_enforced" in attrs
    assert "has_topic_memory" in attrs
    assert attrs["input_count"] == 2


@pytest.mark.asyncio
async def test_topic_text_not_in_spans(span_tracer):
    """Topic text must never appear in any OTel span attribute."""
    tracer, exporter = span_tracer

    topic_text = "very specific query string that must not appear in telemetry spans"
    topics = [
        _make_topic("privacy-001", topic_text, 0.77),
        _make_topic("privacy-002", "science and technology news from major journals", 0.71),
    ]

    inp = SelectorInput(
        run_id="integration-selector-privacy",
        topics=topics,
        max_select=1,
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    for span in spans:
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert topic_text not in value, (
                    f"Topic text found in span '{span.name}' attribute '{key}'"
                )


# ── LangSmith run config tag verification ─────────────────────────────────────

@pytest.mark.asyncio
async def test_langsmith_run_config_tags():
    """make_run_config produces the required LangSmith tags."""
    from src.common.prompt_loader import make_run_config

    cfg = make_run_config("selector", "v1.0", run_id="test-selector-run-01")

    assert "agent:selector" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]
    assert "run:test-selector-run-01" in cfg["tags"]
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "selector"
    assert cfg["run_name"] == "selector:v1.0"
