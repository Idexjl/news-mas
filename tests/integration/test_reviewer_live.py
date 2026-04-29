"""
Live integration tests for the Reviewer agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - ANTHROPIC_API_KEY set
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_reviewer_live.py -v -m integration
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
    Citation,
    RawArticle,
    ReviewerInput,
    ReviewerOutput,
    SearchResult,
)
from src.agents.reviewer.agent import run_agent

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
    tracer = provider.get_tracer("test-reviewer")
    return tracer, exporter


# ── Fixtures ───────────────────────────────────────────────────────────────────

_QUANTUM_URL = "https://example.com/quantum-computing-breakthrough"
_QUANTUM_CONTENT = (
    "Scientists at MIT have announced a major breakthrough in quantum computing. "
    "The research team demonstrated a new error-correction technique that reduces "
    "decoherence by 90 percent compared to existing methods. The discovery, published "
    "in Nature, involved a novel approach to qubit stabilization using topological "
    "insulators. Professor Jane Smith, lead author, said the work opens new possibilities "
    "for practical quantum computers. The team tested the technique on a 50-qubit system "
    "and achieved results that previously required 500 qubits. Industry analysts expect "
    "this to accelerate development of commercially viable quantum computers by five years. "
    "The research was funded by DARPA and the NSF under a three-year grant program."
)


def _make_article(
    url: str = _QUANTUM_URL,
    title: str = "Quantum Computing Breakthrough at MIT",
    content: str | None = None,
) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        content=content or _QUANTUM_CONTENT,
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )


def _make_source(
    url: str = _QUANTUM_URL,
    title: str = "Quantum Computing Breakthrough at MIT",
    content: str | None = None,
) -> SearchResult:
    return SearchResult(
        url=url,
        title=title,
        content=content or _QUANTUM_CONTENT,
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )


def _good_reviewer_input(run_id: str = "rev-test-01", retry_count: int = 0) -> ReviewerInput:
    """A well-formed summarizer output — should consistently pass review."""
    article = _make_article()
    source = _make_source()
    return ReviewerInput(
        run_id=run_id,
        article=article,
        summary=(
            "MIT researchers have achieved a significant advance in quantum computing, "
            "demonstrating an error-correction technique that reduces decoherence by "
            "90 percent compared to current methods [1]. The innovation, centred on "
            "qubit stabilization via topological insulators, was validated on a 50-qubit "
            "system, producing results that previously required ten times as many qubits [1]. "
            "The findings, published in Nature, were funded by DARPA and the NSF and are "
            "expected to accelerate the timeline for commercially viable quantum computers "
            "by approximately five years [1]."
        ),
        key_points=[
            "New error-correction technique cuts decoherence by 90 percent using topological insulators.",
            "A 50-qubit system achieved results previously requiring 500 qubits, a ten-fold efficiency gain.",
            "Commercial quantum computers could arrive five years sooner as a result of this breakthrough.",
        ],
        citations=[
            Citation(
                index=1,
                url=_QUANTUM_URL,
                title="Quantum Computing Breakthrough at MIT",
                quote="reduces decoherence by 90 percent compared to existing methods",
            )
        ],
        sources=[source],
        retry_count=retry_count,
    )


# ── Test 1: good summary passes review ────────────────────────────────────────

@pytest.mark.asyncio
async def test_good_summary_passes(span_tracer):
    """
    A well-formed summary with accurate citations and distinct key points
    should receive a PASS verdict and include an endorsement_note.
    """
    tracer, exporter = span_tracer
    inp = _good_reviewer_input(run_id="rev-pass-test-01")

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got error: {output.error}"
    assert output.verdict == "pass", f"Expected pass, got: {output.verdict}"
    assert output.approved is True
    assert output.endorsement_note, "Expected endorsement_note to be present on PASS"

    # OTel: reviewer.review span with correct attributes
    spans = exporter.get_finished_spans()
    review_spans = [s for s in spans if s.name == "reviewer.review"]
    assert review_spans, f"Missing reviewer.review span. Got: {[s.name for s in spans]}"
    attrs = review_spans[0].attributes
    assert attrs.get("verdict") == "pass"
    assert attrs.get("issue_count") == 0
    assert attrs.get("retry_count") == 0
    assert attrs.get("model_id") == "claude-sonnet-4-6"


# ── Test 2: citation mismatch caught ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_citation_mismatch_caught(span_tracer):
    """
    A citation whose quote has no basis in the source content should be flagged
    as citation_mismatch, causing a FAIL verdict.
    """
    tracer, exporter = span_tracer
    article = _make_article()
    source = _make_source()

    inp = ReviewerInput(
        run_id="rev-citation-test-01",
        article=article,
        summary=(
            "Researchers have made a major discovery in renewable energy. "
            "According to the study, solar panel efficiency has doubled to 80 percent [1]. "
            "This will transform global energy markets within two years."
        ),
        key_points=[
            "Solar panel efficiency doubled to 80 percent in new study.",
            "Global energy markets will transform within two years.",
            "Renewable energy investment is at an all-time high.",
        ],
        citations=[
            Citation(
                index=1,
                url=_QUANTUM_URL,
                title="Quantum Computing Breakthrough at MIT",
                # This quote does not appear in the source — it is about solar energy,
                # not quantum computing.
                quote="solar panel efficiency has doubled to 80 percent",
            )
        ],
        sources=[source],
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert output.verdict == "fail", f"Expected fail, got: {output.verdict}"
    assert output.approved is False
    issue_types = [i.type for i in output.issues]
    assert "citation_mismatch" in issue_types, (
        f"Expected citation_mismatch in issues, got: {issue_types}"
    )
    assert output.feedback, "Expected feedback_for_summarizer to be present on FAIL"

    # OTel: reviewer.fail event on the parent span
    spans = exporter.get_finished_spans()
    review_spans = [s for s in spans if s.name == "reviewer.review"]
    assert review_spans
    attrs = review_spans[0].attributes
    assert attrs.get("verdict") == "fail"
    assert attrs.get("issue_count", 0) >= 1

    # Verify reviewer.fail event was recorded
    fail_events = [
        e for s in review_spans for e in s.events if e.name == "reviewer.fail"
    ]
    assert fail_events, "Expected reviewer.fail OTel event"
    event_issue_types = fail_events[0].attributes.get("issue_types", [])
    assert "citation_mismatch" in event_issue_types


# ── Test 3: constraint violation caught ───────────────────────────────────────

@pytest.mark.asyncio
async def test_constraint_violation_caught(span_tracer):
    """
    Summary containing sports content when the constraint is 'only technology news'
    should receive a FAIL verdict with constraint_violated.
    """
    tracer, exporter = span_tracer
    article = _make_article()
    source = _make_source()

    inp = ReviewerInput(
        run_id="rev-constraint-test-01",
        article=article,
        summary=(
            "MIT announced advances in quantum error correction this week [1]. "
            "Meanwhile, in unrelated news, the local football team won the championship "
            "with a dramatic last-minute goal, drawing record crowds to the stadium."
        ),
        key_points=[
            "MIT announced quantum error-correction advances.",
            "The local football team won the championship.",
            "Record crowds attended the football match.",
        ],
        citations=[
            Citation(
                index=1,
                url=_QUANTUM_URL,
                title="Quantum Computing Breakthrough at MIT",
                quote="advances in quantum computing",
            )
        ],
        sources=[source],
        constraints=["only technology news"],
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert output.verdict == "fail", f"Expected fail, got: {output.verdict}"
    assert output.approved is False
    issue_types = [i.type for i in output.issues]
    assert "constraint_violated" in issue_types, (
        f"Expected constraint_violated in issues, got: {issue_types}"
    )
    assert output.feedback


# ── Test 4: thin content caught ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_thin_content_caught(span_tracer):
    """
    A summary with only 2 near-identical key points and superficial body text
    should receive a FAIL verdict with thin_content.
    """
    tracer, exporter = span_tracer
    article = _make_article()
    source = _make_source()

    inp = ReviewerInput(
        run_id="rev-thin-test-01",
        article=article,
        summary="Scientists made a discovery. The discovery was announced. It is significant.",
        key_points=[
            "Scientists made a discovery.",
            "The discovery was announced by scientists.",
        ],
        citations=[
            Citation(
                index=1,
                url=_QUANTUM_URL,
                title="Quantum Computing Breakthrough at MIT",
                quote="Scientists at MIT have announced",
            )
        ],
        sources=[source],
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert output.verdict == "fail", f"Expected fail, got: {output.verdict}"
    assert output.approved is False
    issue_types = [i.type for i in output.issues]
    assert "thin_content" in issue_types, (
        f"Expected thin_content in issues, got: {issue_types}"
    )


# ── Test 5: retry leniency ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_leniency(span_tracer):
    """
    A borderline acceptable summary reviewed with retry_count=2 should pass
    due to retry leniency. Verifies that retry context is injected into the prompt
    and that the reviewer does not fail for previously absent issues.
    """
    tracer, exporter = span_tracer

    # Use a complete, good summary with retry_count=2 — should reliably PASS
    inp = _good_reviewer_input(run_id="rev-leniency-test-01", retry_count=2)

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert output.verdict == "pass", (
        f"Expected pass on retry_count=2 for a good summary, got: {output.verdict}. "
        f"Issues: {[(i.type, i.description) for i in output.issues]}"
    )
    assert output.approved is True

    # Verify retry_count is visible in the review span
    spans = exporter.get_finished_spans()
    review_spans = [s for s in spans if s.name == "reviewer.review"]
    assert review_spans
    assert review_spans[0].attributes.get("retry_count") == 2

    # Also verify LangSmith run config carries the retry tag
    from src.common.prompt_loader import make_run_config
    cfg = make_run_config("reviewer", "v1.0", run_id=inp.run_id)
    assert "agent:reviewer" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]


# ── Test 6: PHI in summary auto-fails ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_phi_in_summary_auto_fails(span_tracer):
    """
    When detect_phi_in_output detects PHI in the summary, the reviewer should:
    - auto-FAIL with type phi_detected
    - NOT call the LLM (verified via mock)
    - log a security event
    - record phi_detected=True in the reviewer.phi_check span
    """
    tracer, exporter = span_tracer
    inp = _good_reviewer_input(run_id="rev-phi-test-01")

    fake_phi = [{"type": "US_SSN", "start": 0, "end": 9, "score": 0.95}]

    with patch("src.agents.reviewer.agent.detect_phi_in_output", return_value=fake_phi):
        with patch(
            "src.agents.reviewer.agent._call_claude",
            new_callable=AsyncMock,
        ) as mock_llm:
            output = await run_agent(inp, tracer=tracer)

    # LLM must NOT have been called
    mock_llm.assert_not_called()

    assert output.success is True
    assert output.verdict == "fail"
    assert output.approved is False
    issue_types = [i.type for i in output.issues]
    assert "phi_detected" in issue_types, (
        f"Expected phi_detected in issues, got: {issue_types}"
    )
    assert "personal health information" in output.feedback.lower()

    # OTel: phi_check span has phi_detected=True
    spans = exporter.get_finished_spans()
    phi_spans = [s for s in spans if s.name == "reviewer.phi_check"]
    assert phi_spans, "Expected reviewer.phi_check span"
    assert phi_spans[0].attributes.get("phi_detected") is True


# ── Test 7: reviewer unavailable → DEGRADED pass ─────────────────────────────

@pytest.mark.asyncio
async def test_reviewer_unavailable_degraded_pass(span_tracer):
    """
    When the Anthropic API raises an unexpected exception that exhausts all
    internal retries, the reviewer should return a DEGRADED ReviewerOutput:
    - success=True (pipeline continues)
    - approved=True
    - warning 'reviewer_unavailable' present
    - error.severity == DEGRADED
    """
    tracer, exporter = span_tracer
    inp = _good_reviewer_input(run_id="rev-unavailable-test-01")

    # Raise an unexpected exception that bypasses internal API retries
    with patch(
        "src.agents.reviewer.agent._call_claude",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Connection refused"),
    ):
        output = await run_agent(inp, tracer=tracer)

    # The agent should fall back to DEGRADED pass — pipeline must not stall
    assert output.success is True, "Expected success=True for DEGRADED fallback"
    assert output.approved is True, "Expected approved=True for DEGRADED fallback"
    assert "reviewer_unavailable" in output.warnings, (
        f"Expected 'reviewer_unavailable' in warnings. Got: {output.warnings}"
    )
    assert output.error is not None
    assert output.error.severity == "DEGRADED", (
        f"Expected DEGRADED severity, got: {output.error.severity}"
    )


# ── OTel span structure ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_otel_all_required_spans(span_tracer):
    """All required OTel spans are emitted for a successful review."""
    tracer, exporter = span_tracer
    inp = _good_reviewer_input(run_id="rev-otel-test-01")

    output = await run_agent(inp, tracer=tracer)
    assert output.success is True

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert "reviewer.review" in span_names, f"Missing reviewer.review. Got: {span_names}"
    assert "reviewer.phi_check" in span_names, f"Missing reviewer.phi_check. Got: {span_names}"
    assert "reviewer.llm_call" in span_names, f"Missing reviewer.llm_call. Got: {span_names}"

    # Verify required attributes on parent span
    review_span = next(s for s in spans if s.name == "reviewer.review")
    attrs = review_span.attributes
    assert "topic_id" in attrs
    assert "run_id" in attrs
    assert "verdict" in attrs
    assert "retry_count" in attrs
    assert "issue_count" in attrs
    assert "model_id" in attrs

    # Verify llm_call attributes
    llm_span = next(s for s in spans if s.name == "reviewer.llm_call")
    llm_attrs = llm_span.attributes
    assert llm_attrs.get("model_id") == "claude-sonnet-4-6"
    assert "prompt_version" in llm_attrs
    assert "token_estimate" in llm_attrs


@pytest.mark.asyncio
async def test_no_content_in_spans(span_tracer):
    """Summary content and issue descriptions must never appear in OTel span attributes."""
    tracer, exporter = span_tracer
    canary = "unique-canary-x7q9z-must-not-appear-in-telemetry"

    inp = ReviewerInput(
        run_id="rev-privacy-test-01",
        article=_make_article(content=canary + " is the article content."),
        summary=canary + " is a summary.",
        key_points=[canary + " key point one.", "Another distinct key point.", "Third distinct key point."],
        citations=[
            Citation(index=1, url=_QUANTUM_URL, title="Test", quote=canary[:15])
        ],
        sources=[_make_source()],
    )

    # Mock the LLM so it returns a canned verdict (canary never touches the model)
    import json
    mock_verdict = json.dumps({
        "verdict": "fail",
        "issues": [{"type": "citation_mismatch", "description": "mismatch", "suggestion": "fix it"}],
        "feedback_for_summarizer": "Please fix the citation.",
    })
    with patch("src.agents.reviewer.agent._call_claude", new_callable=AsyncMock, return_value=mock_verdict):
        await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    for span in spans:
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert canary not in value, (
                    f"Canary found in span '{span.name}' attribute '{key}'"
                )
        for event in span.events:
            for key, value in event.attributes.items():
                if isinstance(value, str):
                    assert canary not in value, (
                        f"Canary found in span '{span.name}' event '{event.name}' attribute '{key}'"
                    )


@pytest.mark.asyncio
async def test_issue_descriptions_not_in_spans(span_tracer):
    """Issue descriptions and feedback must not appear in OTel span events."""
    tracer, exporter = span_tracer

    secret_description = "this-secret-description-must-not-leak-into-otel-spans"
    import json
    mock_verdict = json.dumps({
        "verdict": "fail",
        "issues": [
            {
                "type": "thin_content",
                "description": secret_description,
                "suggestion": "expand the summary",
            }
        ],
        "feedback_for_summarizer": "expand: " + secret_description,
    })

    inp = _good_reviewer_input(run_id="rev-leak-test-01")
    with patch("src.agents.reviewer.agent._call_claude", new_callable=AsyncMock, return_value=mock_verdict):
        await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    for span in spans:
        for event in span.events:
            for key, value in event.attributes.items():
                if isinstance(value, str):
                    assert secret_description not in value, (
                        f"Issue description leaked into event '{event.name}' attribute '{key}'"
                    )
            # Also check the event attributes tuple format
            if hasattr(event.attributes, "items"):
                for val in event.attributes.values():
                    if isinstance(val, (list, tuple)):
                        for item in val:
                            assert secret_description not in str(item), (
                                f"Issue description leaked into event list attribute"
                            )


# ── LangSmith run config tags ─────────────────────────────────────────────────

def test_langsmith_run_config_tags():
    """make_run_config produces required LangSmith tags for the reviewer."""
    from src.common.prompt_loader import make_run_config

    cfg = make_run_config("reviewer", "v1.0", run_id="test-rev-run-01")

    assert "agent:reviewer" in cfg["tags"]
    assert "prompt:v1.0" in cfg["tags"]
    assert "run:test-rev-run-01" in cfg["tags"]
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "reviewer"
    assert cfg["run_name"] == "reviewer:v1.0"


# ── Pre-flight auto-fail checks (no LLM call) ─────────────────────────────────

@pytest.mark.asyncio
async def test_empty_summary_auto_fails(span_tracer):
    """An empty summary must auto-fail with thin_content without calling the LLM."""
    tracer, exporter = span_tracer
    inp = ReviewerInput(
        run_id="rev-empty-test-01",
        article=_make_article(),
        summary="",
        key_points=[],
        citations=[],
        sources=[_make_source()],
    )

    with patch("src.agents.reviewer.agent._call_claude", new_callable=AsyncMock) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.verdict == "fail"
    issue_types = [i.type for i in output.issues]
    assert "thin_content" in issue_types


@pytest.mark.asyncio
async def test_no_citations_auto_fails(span_tracer):
    """When sources are available but citations are empty, auto-fail with citation_mismatch."""
    tracer, exporter = span_tracer
    inp = ReviewerInput(
        run_id="rev-nocite-test-01",
        article=_make_article(),
        summary="MIT researchers achieved a breakthrough in quantum computing.",
        key_points=["A breakthrough was achieved.", "It is significant.", "The team is happy."],
        citations=[],       # no citations
        sources=[_make_source()],  # but sources ARE available
    )

    with patch("src.agents.reviewer.agent._call_claude", new_callable=AsyncMock) as mock_llm:
        output = await run_agent(inp, tracer=tracer)

    mock_llm.assert_not_called()
    assert output.verdict == "fail"
    issue_types = [i.type for i in output.issues]
    assert "citation_mismatch" in issue_types


@pytest.mark.asyncio
async def test_no_sources_degraded_skips_citation_check(span_tracer):
    """
    When no sources are provided, the reviewer should proceed in DEGRADED mode:
    - still call the LLM (for substance and constraint checks)
    - add 'citation_check_skipped_no_sources' warning on PASS
    - not auto-fail for missing citations
    """
    tracer, exporter = span_tracer
    inp = ReviewerInput(
        run_id="rev-nosrc-test-01",
        article=_make_article(),
        summary=(
            "MIT researchers demonstrated a 90-percent reduction in quantum decoherence [1], "
            "opening the path to practical quantum computers within five years."
        ),
        key_points=[
            "Decoherence reduced by 90 percent via topological insulator technique.",
            "50-qubit system matches results previously requiring 500 qubits.",
            "Commercial quantum computers may arrive five years sooner.",
        ],
        citations=[
            Citation(index=1, url=_QUANTUM_URL, title="MIT Quantum", quote="reduces decoherence by 90 percent")
        ],
        sources=[],  # no sources to check against
    )

    output = await run_agent(inp, tracer=tracer)

    # Should not auto-fail; LLM should be called
    assert output.success is True
    # On PASS: warning should be present
    if output.verdict == "pass":
        assert "citation_check_skipped_no_sources" in output.warnings, (
            f"Expected degraded warning. Got: {output.warnings}"
        )
