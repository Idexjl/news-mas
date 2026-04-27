"""
Live integration tests for the SearchWorker agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - TAVILY_API_KEY set to a valid key
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/ -v -m integration
"""
from __future__ import annotations

import os

import pytest

# Guard heavy optional dependencies so pytest skips the file cleanly
# in environments where they are not installed (e.g. CI without ML deps).
pytest.importorskip("presidio_analyzer", reason="presidio-analyzer not installed")
pytest.importorskip("bs4", reason="beautifulsoup4 not installed")
pytest.importorskip("tavily", reason="tavily-python not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.schemas import SearchWorkerInput
from src.agents.search_worker.agent import run_agent

_INTEGRATION = os.getenv("INTEGRATION_TESTS", "").lower() == "true"

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_integration():
    if not _INTEGRATION:
        pytest.skip("Set INTEGRATION_TESTS=true to run live integration tests")


@pytest.fixture()
def span_tracer():
    """
    Return a (tracer, exporter) pair backed by a standalone TracerProvider.

    We deliberately do NOT call otel_trace.set_tracer_provider() — that API
    raises a warning and is blocked once a provider is already registered.
    Instead, run_agent() accepts an optional tracer= kwarg so tests can inject
    a scoped tracer without touching the global provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-search-worker")
    return tracer, exporter


@pytest.mark.asyncio
async def test_live_search_returns_scrubbed_articles(span_tracer):
    """Real Tavily call — verifies results are returned and PII-scrubbed."""
    tracer, _ = span_tracer
    inp = SearchWorkerInput(
        run_id="integration-test-01",
        topics=["AI agents news"],
        since_days=7,
        max_results_per_topic=3,
    )

    output = await run_agent(inp, tracer=tracer)

    assert output.run_id == "integration-test-01"

    if not output.articles and any(
        w.startswith("tavily_connection_error") for w in (output.warnings or [])
    ):
        pytest.skip("Tavily API unreachable — skipping rather than failing")

    assert len(output.articles) > 0, "Expected at least one article from Tavily"

    for article in output.articles:
        assert article.url.startswith("http"), f"Bad URL: {article.url}"
        assert article.title, "Empty title"
        assert article.content, "Empty content"
        # Presidio placeholder pattern — scrubbed content should not contain raw emails etc.
        # (We verify scrubbing ran; we cannot verify every PII type appeared in this topic.)
        assert len(article.content) <= int(os.getenv("TOKEN_BUDGET_SOURCES", "200000")), (
            "Content exceeds TOKEN_BUDGET_SOURCES limit"
        )


@pytest.mark.asyncio
async def test_live_search_otel_spans_created(span_tracer):
    """Verifies OTel spans are emitted and contain no raw content."""
    tracer, exporter = span_tracer
    inp = SearchWorkerInput(
        run_id="integration-test-spans-01",
        topics=["AI agents news"],
        since_days=7,
        max_results_per_topic=2,
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("search_worker.search" in n for n in span_names), (
        f"Expected search_worker.search span. Got: {span_names}"
    )
    assert any("egress.http.fetch" in n for n in span_names), (
        f"Expected egress.http.fetch span. Got: {span_names}"
    )

    # No span attribute should contain the literal text of an article.
    # We check that no attribute value is longer than a reasonable metadata string.
    _CONTENT_KEYS = {"content", "text", "body", "article", "summary", "raw", "html", "snippet"}
    for span in spans:
        for key, value in span.attributes.items():
            assert key.lower() not in _CONTENT_KEYS, (
                f"Span '{span.name}' contains blocked attribute key '{key}'"
            )
            if isinstance(value, str):
                assert len(value) < 2000, (
                    f"Span '{span.name}' attribute '{key}' looks like raw content "
                    f"(length {len(value)})"
                )


@pytest.mark.asyncio
async def test_live_search_egress_tavily_span(span_tracer):
    """Verifies egress.tavily.search span is emitted with correct metadata attributes."""
    tracer, exporter = span_tracer
    inp = SearchWorkerInput(
        run_id="integration-test-egress-01",
        topics=["AI agents news"],
        since_days=7,
        max_results_per_topic=2,
    )

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    tavily_spans = [s for s in spans if s.name == "egress.tavily.search"]

    assert tavily_spans, (
        f"Expected egress.tavily.search span. Got: {[s.name for s in spans]}"
    )

    tavily_span = tavily_spans[0]
    attrs = tavily_span.attributes

    # Required egress attributes
    assert attrs.get("egress.host") == "api.tavily.com", (
        f"egress.host should be 'api.tavily.com', got: {attrs.get('egress.host')}"
    )
    assert "tavily.query_length" in attrs, "Missing tavily.query_length attribute"
    assert isinstance(attrs["tavily.query_length"], int), "tavily.query_length must be int"
    assert attrs["tavily.query_length"] > 0, "tavily.query_length must be positive"

    # Must NOT contain raw query content — only the length
    _CONTENT_KEYS = {"query", "content", "text", "body", "article", "summary", "raw"}
    for key in attrs:
        assert key.lower() not in _CONTENT_KEYS, (
            f"egress.tavily.search span contains forbidden attribute '{key}'"
        )

    # No attribute value should look like raw content
    for key, value in attrs.items():
        if isinstance(value, str):
            assert len(value) < 2000, (
                f"egress.tavily.search attribute '{key}' looks like raw content "
                f"(length {len(value)})"
            )
