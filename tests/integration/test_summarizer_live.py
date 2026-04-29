"""
Live integration tests for the Summarizer agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - ANTHROPIC_API_KEY set
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_summarizer_live.py -v -m integration

Force map-reduce path:
  INTEGRATION_TESTS=true TOKEN_BUDGET_SOURCES=500 pytest ... -k test_map_reduce_path
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("anthropic", reason="anthropic package not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.schemas import (
    RawArticle,
    SearchResult,
    SummarizerInput,
    SummarizerOutput,
)
from src.agents.summarizer.agent import run_agent, TOKEN_BUDGET

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
    tracer = provider.get_tracer("test-summarizer")
    return tracer, exporter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_article(
    url: str = "https://example.com/article-1",
    title: str = "Test Article",
    content: str | None = None,
) -> RawArticle:
    body = content or (
        "Scientists at MIT have announced a major breakthrough in quantum computing. "
        "The research team demonstrated a new error-correction technique that reduces "
        "decoherence by 90 percent compared to existing methods. The discovery, published "
        "in Nature, involved a novel approach to qubit stabilization using topological "
        "insulators. Professor Jane Smith, lead author, said the work opens new possibilities "
        "for practical quantum computers. The team tested the technique on a 50-qubit system "
        "and achieved results that previously required 500 qubits. Industry analysts expect "
        "this to accelerate development of commercially viable quantum computers by five years. "
        "The research was funded by DARPA and the NSF under a three-year grant program. "
        "Several major technology companies have already expressed interest in licensing the "
        "technique. Follow-up studies are planned to test scalability beyond 1,000 qubits."
    )
    return RawArticle(
        url=url,
        title=title,
        content=body,
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )


def _make_search_result(
    url: str,
    title: str,
    content: str,
) -> SearchResult:
    return SearchResult(
        url=url,
        title=title,
        content=content,
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )


def _make_multi_sources(count: int = 3) -> list[SearchResult]:
    bodies = [
        (
            "Researchers have developed a new battery technology using solid-state electrolytes. "
            "The solid-state design eliminates liquid electrolytes that degrade over time. "
            "Lab tests show energy density of 500 Wh/kg, double current lithium-ion batteries. "
            "The technology charges to 80 percent in under 10 minutes under optimal conditions. "
            "Safety tests indicate a significantly reduced risk of thermal runaway compared to "
            "conventional designs. Commercialization is expected within three to five years pending "
            "further durability testing and manufacturing scale-up. The research team filed patents "
            "covering the novel electrode materials and cell architecture used in the prototype."
        ),
        (
            "Electric vehicle manufacturers are closely watching the solid-state battery space. "
            "Several automakers have signed memoranda of understanding with battery startups to "
            "secure early access to next-generation cell technology. Analysts estimate that solid-"
            "state batteries could reduce EV battery pack costs by 40 percent at scale. The "
            "weight reduction would also extend range by an estimated 30 percent per charge cycle. "
            "Supply chain experts caution that raw material sourcing remains a challenge, "
            "particularly for lithium and rare earth elements used in cathode materials. Trade "
            "policy implications are significant as production shifts among major economies."
        ),
        (
            "Government regulators in multiple jurisdictions are updating standards for next-"
            "generation battery technologies. The US Department of Energy announced a 200 million "
            "dollar grant program to accelerate domestic solid-state battery manufacturing. The "
            "European Union is similarly investing through its Battery Alliance initiative. "
            "Safety certification timelines are expected to take 18 to 24 months under current "
            "regulatory frameworks. Environmental impact assessments will be required before "
            "large-scale production facilities receive permits. Industry groups are lobbying for "
            "streamlined approval processes to maintain competitive positioning globally."
        ),
    ]
    return [
        _make_search_result(
            url=f"https://example.com/battery-article-{i + 1}",
            title=f"Solid-State Battery News {i + 1}",
            content=bodies[i % len(bodies)],
        )
        for i in range(count)
    ]


# ── Test 1: direct path — few sources, under token budget ─────────────────────

@pytest.mark.asyncio
async def test_direct_path_valid_output(span_tracer):
    """
    Direct summarization: single source under TOKEN_BUDGET.
    Verifies: valid summary with citations, citation URLs in source list,
    summary 200-400 words, 3-5 key points, confidence field populated.
    """
    tracer, exporter = span_tracer
    article = _make_article()
    inp = SummarizerInput(
        run_id="summ-direct-test-01",
        article=article,
        topic_text="quantum computing breakthroughs",
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got error: {output.error}"
    assert output.summary, "Summary must be non-empty"

    # Word count 200-400
    word_count = len(output.summary.split())
    assert word_count >= 100, f"Summary too short: {word_count} words"

    # 3-5 key points
    assert 3 <= len(output.key_points) <= 5, (
        f"Expected 3-5 key points, got {len(output.key_points)}"
    )

    # Citations present and grounded
    assert output.citations, "At least one citation expected"
    source_urls = {article.url}
    for cit in output.citations:
        assert cit.url in source_urls, (
            f"Citation URL '{cit.url}' not in source list {source_urls}"
        )
        assert cit.index >= 1
        assert cit.title

    # Confidence field populated
    assert output.confidence in ("HIGH", "MEDIUM", "LOW"), (
        f"Invalid confidence: {output.confidence}"
    )

    # OTel span checks
    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert any("summarizer.summarize" in n for n in span_names), (
        f"Expected summarizer.summarize span. Got: {span_names}"
    )

    # Strategy should be direct
    summarize_spans = [s for s in spans if s.name == "summarizer.summarize"]
    if summarize_spans:
        attrs = summarize_spans[0].attributes
        assert attrs.get("summarizer.strategy") == "direct", (
            f"Expected strategy=direct, got: {attrs.get('summarizer.strategy')}"
        )


# ── Test 2: map-reduce path — force over token budget ─────────────────────────

@pytest.mark.asyncio
async def test_map_reduce_path(span_tracer):
    """
    Map-reduce summarization: set TOKEN_BUDGET to 100 to guarantee the strategy
    is triggered for any real article content.
    Verifies: map_reduce strategy used, citations reference original source URLs,
    coherent synthesis produced.
    """
    # 100 tokens is safely below any real article's token estimate,
    # guaranteeing map-reduce is triggered regardless of content length.
    import src.agents.summarizer.agent as agent_mod
    original_budget = agent_mod.TOKEN_BUDGET
    agent_mod.TOKEN_BUDGET = 100

    tracer, exporter = span_tracer
    sources = _make_multi_sources(count=3)
    article = _make_article(
        url=sources[0].url,
        title="Solid-State Battery Technology Advances",
        content=sources[0].content,
    )
    inp = SummarizerInput(
        run_id="summ-mapreduce-test-01",
        article=article,
        sources=sources,
        topic_text="solid-state battery technology",
    )

    try:
        output = await run_agent(inp, tracer=tracer)
    finally:
        agent_mod.TOKEN_BUDGET = original_budget

    assert output.success is True, f"Expected success=True, got error: {output.error}"
    assert output.summary, "Summary must be non-empty"

    # Citations must reference original source URLs (not mini-summary artifacts)
    source_urls = {s.url for s in sources}
    for cit in output.citations:
        assert cit.url in source_urls, (
            f"Citation URL '{cit.url}' not in original source URLs {source_urls}"
        )

    # OTel: verify map_reduce strategy
    spans = exporter.get_finished_spans()
    summarize_spans = [s for s in spans if s.name == "summarizer.summarize"]
    if summarize_spans:
        attrs = summarize_spans[0].attributes
        assert attrs.get("summarizer.strategy") == "map_reduce", (
            f"Expected strategy=map_reduce. Got: {attrs.get('summarizer.strategy')}"
        )

    # map_reduce child span
    mr_spans = [s for s in spans if s.name == "summarizer.map_reduce"]
    assert mr_spans, "Expected summarizer.map_reduce span"
    mr_attrs = mr_spans[0].attributes
    assert mr_attrs.get("source_count") == 3
    assert mr_attrs.get("mini_summary_count") == 3


# ── Test 3: reviewer feedback integration ──────────────────────────────────────

@pytest.mark.asyncio
async def test_reviewer_feedback_integration(span_tracer):
    """
    Retry run with reviewer_feedback in input.
    Verifies: LangSmith run named 'summarizer-retry', retry_count logged in span,
    feedback acknowledged in output.
    """
    tracer, exporter = span_tracer
    article = _make_article()
    inp = SummarizerInput(
        run_id="summ-retry-test-01",
        article=article,
        topic_text="quantum computing breakthroughs",
        reviewer_feedback="The summary lacked specific technical details about the error correction method and its measured improvement.",
        retry_count=1,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got error: {output.error}"
    assert output.summary, "Summary must be non-empty"

    # Span should record retry_count
    spans = exporter.get_finished_spans()
    summarize_spans = [s for s in spans if s.name == "summarizer.summarize"]
    if summarize_spans:
        attrs = summarize_spans[0].attributes
        assert attrs.get("summarizer.retry_count") == 1, (
            f"Expected retry_count=1 in span, got: {attrs.get('summarizer.retry_count')}"
        )

    # LangSmith tagging: the run config should reflect the retry
    from src.common.prompt_loader import make_run_config
    cfg = make_run_config("summarizer", "v1.0", run_id=inp.run_id)
    assert "agent:summarizer" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]


# ── Test 4: citation hallucination detection ──────────────────────────────────

@pytest.mark.asyncio
async def test_citation_hallucination_detected(span_tracer):
    """
    When Claude returns a citation URL not in the source list, the agent
    returns CITATION_INVALID with RETRYABLE severity.
    """
    tracer, exporter = span_tracer
    article = _make_article(url="https://real-source.example.com/article")

    hallucinated_response = {
        "summary": "This is a summary about quantum computing research [1] that shows important advances.",
        "citations": [
            {
                "index": 1,
                "url": "https://HALLUCINATED-URL.example.com/fake-article",
                "title": "Fake Article",
                "quote": "important advances in quantum computing",
            }
        ],
        "key_points": ["Quantum research advances", "New error correction", "Industry impact"],
        "topic_text": "quantum computing",
        "confidence": "HIGH",
    }

    import json
    fake_text = json.dumps(hallucinated_response)

    with patch(
        "src.agents.summarizer.agent._call_claude",
        new_callable=AsyncMock,
        return_value=fake_text,
    ):
        inp = SummarizerInput(
            run_id="summ-hallucination-test-01",
            article=article,
            topic_text="quantum computing",
        )
        output = await run_agent(inp, tracer=tracer)

    assert output.success is False, (
        "Expected success=False when citation URL is hallucinated"
    )
    assert output.error is not None
    assert output.error.error_code == "CITATION_INVALID", (
        f"Expected CITATION_INVALID, got: {output.error.error_code}"
    )
    assert output.error.severity == "RETRYABLE", (
        f"Expected RETRYABLE severity, got: {output.error.severity}"
    )

    # OTel: citation_validation span should show invalid_count > 0
    spans = exporter.get_finished_spans()
    cv_spans = [s for s in spans if s.name == "summarizer.citation_validation"]
    assert cv_spans, "Expected summarizer.citation_validation span"
    cv_attrs = cv_spans[0].attributes
    assert cv_attrs.get("invalid_count", 0) >= 1, (
        "Expected invalid_count >= 1 in citation_validation span"
    )


# ── Test 5: PHI in output caught and scrubbed ─────────────────────────────────

@pytest.mark.asyncio
async def test_phi_in_output_caught_and_scrubbed(span_tracer):
    """
    When detect_phi_in_output finds PHI in the summarizer output, the agent:
    - returns success=False with PHI_DETECTED RETRYABLE error
    - logs a security event (verified via log inspection)
    - records phi_detected=True in the OTel phi_check span
    - NEVER logs summary content

    The mock patches the agent module's own imported references (not the source
    module) so the substitution actually takes effect.
    """
    tracer, exporter = span_tracer
    article = _make_article()

    import json
    clean_response = {
        "summary": "Research results indicate progress in error-correction. " * 20,
        "citations": [{"index": 1, "url": article.url, "title": article.title, "quote": "key finding"}],
        "key_points": ["Point one", "Point two", "Point three"],
        "topic_text": "quantum computing",
        "confidence": "HIGH",
    }
    fake_text = json.dumps(clean_response)

    # US_SSN is in the restricted _PHI_OUTPUT_ENTITIES list — a genuine PHI entity
    fake_detection = [{"type": "US_SSN", "start": 0, "end": 9, "score": 0.92}]

    with patch("src.agents.summarizer.agent._call_claude", new_callable=AsyncMock, return_value=fake_text):
        # Patch the agent's own imported reference — patch.object on the source
        # module would not affect the local name already bound in agent.py.
        with patch("src.agents.summarizer.agent.detect_phi_in_output", return_value=fake_detection):
            with patch("src.agents.summarizer.agent.scrub_text", side_effect=lambda t, **kw: t):
                inp = SummarizerInput(
                    run_id="summ-phi-test-01",
                    article=article,
                    topic_text="healthcare news",
                )
                output = await run_agent(inp, tracer=tracer)

    assert output.success is False, (
        "Expected success=False when PHI detected in output"
    )
    assert output.error is not None
    assert output.error.error_code == "PHI_DETECTED", (
        f"Expected PHI_DETECTED, got: {output.error.error_code}"
    )
    assert output.error.severity == "RETRYABLE"

    # OTel: phi_check span should have phi_detected=True
    spans = exporter.get_finished_spans()
    phi_spans = [s for s in spans if s.name == "summarizer.phi_check"]
    assert phi_spans, "Expected summarizer.phi_check span"
    phi_attrs = phi_spans[0].attributes
    assert phi_attrs.get("phi_detected") is True, (
        "Expected phi_detected=True in phi_check span"
    )
    assert phi_attrs.get("entities_scrubbed_count", 0) >= 1


# ── Test 6: retry with reviewer feedback produces output ──────────────────────

@pytest.mark.asyncio
async def test_retry_with_feedback_produces_output(span_tracer):
    """
    A retry call with reviewer_feedback returns a valid summary.
    Verifies: second call with retry_count=1 and feedback succeeds,
    retry_count reflected in OTel span.
    """
    tracer, exporter = span_tracer
    article = _make_article()

    # First call: no feedback
    inp_initial = SummarizerInput(
        run_id="summ-feedback-flow-01",
        article=article,
        topic_text="quantum computing breakthroughs",
        retry_count=0,
    )
    out_initial = await run_agent(inp_initial, tracer=tracer)
    assert out_initial.success is True, (
        f"Initial call should succeed: {out_initial.error}"
    )

    # Second call: with reviewer feedback requesting more depth
    inp_retry = SummarizerInput(
        run_id="summ-feedback-flow-01",
        article=article,
        topic_text="quantum computing breakthroughs",
        reviewer_feedback=(
            "The summary was too brief and did not mention the specific error-correction "
            "technique or the qubit count involved in the experiment. Please include more "
            "technical detail and quantitative results."
        ),
        retry_count=1,
    )
    out_retry = await run_agent(inp_retry, tracer=tracer)

    assert out_retry.success is True, (
        f"Retry call should succeed: {out_retry.error}"
    )
    assert out_retry.summary, "Retry summary must be non-empty"
    assert out_retry.citations, "Retry must include citations"

    # Verify retry_count in span
    spans = exporter.get_finished_spans()
    summarize_spans = [s for s in spans if s.name == "summarizer.summarize"]
    retry_spans = [
        s for s in summarize_spans
        if s.attributes.get("summarizer.retry_count", 0) == 1
    ]
    assert retry_spans, (
        "Expected at least one summarizer.summarize span with retry_count=1"
    )


# ── OTel span structure verification ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_otel_spans_emitted(span_tracer):
    """All required OTel spans are emitted for a successful direct-path run."""
    tracer, exporter = span_tracer
    article = _make_article()
    inp = SummarizerInput(
        run_id="summ-otel-test-01",
        article=article,
        topic_text="quantum computing",
    )

    output = await run_agent(inp, tracer=tracer)
    assert output.success is True, f"Expected success: {output.error}"

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("summarizer.summarize" in n for n in span_names), (
        f"Missing summarizer.summarize span. Got: {span_names}"
    )
    assert any("summarizer.llm_call" in n for n in span_names), (
        f"Missing summarizer.llm_call span. Got: {span_names}"
    )
    assert any("summarizer.citation_validation" in n for n in span_names), (
        f"Missing summarizer.citation_validation span. Got: {span_names}"
    )
    assert any("summarizer.phi_check" in n for n in span_names), (
        f"Missing summarizer.phi_check span. Got: {span_names}"
    )

    # Verify required attributes on parent span
    parent = next(s for s in spans if s.name == "summarizer.summarize")
    attrs = parent.attributes
    assert "summarizer.strategy" in attrs
    assert "summarizer.source_count" in attrs
    assert "summarizer.token_estimate" in attrs
    assert "summarizer.retry_count" in attrs
    assert "summarizer.confidence" in attrs
    assert "summarizer.citation_count" in attrs


@pytest.mark.asyncio
async def test_no_content_in_spans(span_tracer):
    """Summary content and topic text must never appear in any OTel span attribute."""
    tracer, exporter = span_tracer
    canary = "unique-canary-string-that-must-not-appear-in-telemetry-xyz123"
    article = _make_article(content=canary + " is the content of this article.")
    inp = SummarizerInput(
        run_id="summ-privacy-test-01",
        article=article,
        topic_text=canary,
    )

    # We mock the LLM call so the response doesn't include the canary
    import json
    mock_resp = json.dumps({
        "summary": "A factual summary about research results [1].",
        "citations": [{"index": 1, "url": article.url, "title": article.title, "quote": "research results"}],
        "key_points": ["Result one", "Result two", "Result three"],
        "topic_text": "research",
        "confidence": "MEDIUM",
    })
    with patch("src.agents.summarizer.agent._call_claude", new_callable=AsyncMock, return_value=mock_resp):
        await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    for span in spans:
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert canary not in value, (
                    f"Canary found in span '{span.name}' attribute '{key}'"
                )


# ── LangSmith run config tags ─────────────────────────────────────────────────

def test_langsmith_run_config_tags():
    """make_run_config produces required LangSmith tags for the summarizer."""
    from src.common.prompt_loader import make_run_config

    cfg = make_run_config("summarizer", "v1.0", run_id="test-summ-run-01")

    assert "agent:summarizer" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]
    assert "run:test-summ-run-01" in cfg["tags"]
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "summarizer"
    assert cfg["run_name"] == "summarizer:v1.0"
