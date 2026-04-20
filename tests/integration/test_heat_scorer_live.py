"""
Live integration tests for the HeatScorer agent.

Requirements:
  - INTEGRATION_TESTS=true in environment
  - Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434)
  - gemma4:e4b model pulled: ollama pull gemma4:e4b
  - MAS_SECRET_KEY set (or left unset for no-op validation)

Run with:
  INTEGRATION_TESTS=true pytest tests/integration/test_heat_scorer_live.py -v -m integration
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pytest.importorskip("ollama", reason="ollama package not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.prompt_loader import make_run_config
from src.common.schemas import HeatScorerInput, RawArticle
from src.agents.heat_scorer.agent import run_agent

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
    We do NOT call otel_trace.set_tracer_provider() — instead run_agent
    accepts an optional tracer= kwarg so tests inject a scoped tracer without
    touching the global provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-heat-scorer")
    return tracer, exporter


@pytest.fixture()
def ai_article() -> RawArticle:
    """Synthetic AI news article — no PHI, no injection attempts."""
    return RawArticle(
        url="https://example.com/ai-reasoning-model",
        title="Major AI Lab Releases Next-Generation Reasoning Model",
        content=(
            "A leading artificial intelligence laboratory announced today the release "
            "of a new reasoning model that significantly outperforms previous systems "
            "on mathematical and scientific benchmarks. The model is available via API "
            "and has attracted widespread coverage from technology media. Multiple "
            "independent evaluations confirm the benchmark improvements. Industry "
            "analysts expect the release to accelerate adoption in enterprise settings."
        ),
        published_at=datetime.now(timezone.utc),
        source="TechNews",
    )


# ── Core scoring tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_heat_scorer_returns_valid_score(span_tracer, ai_article):
    """Real Gemma 4 call — heat_score is in [0.0, 1.0] and reasoning is non-empty."""
    tracer, _ = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-01", articles=[ai_article])

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True, f"Expected success=True, got error: {output.error}"
    assert output.run_id == "integration-heat-01"
    assert len(output.scored_articles) == 1

    scored = output.scored_articles[0]
    assert 0.0 <= scored.heat_score <= 1.0, (
        f"heat_score {scored.heat_score} is outside [0.0, 1.0]"
    )
    assert scored.reasoning, "reasoning must be a non-empty string"


@pytest.mark.asyncio
async def test_live_heat_scorer_four_signals_present(span_tracer):
    """Verifies all four signal components are returned — uses a direct JSON parse."""
    tracer, _ = span_tracer
    import json
    import ollama as _ollama
    from src.common.prompt_loader import get_system_prompt, load_prompt

    prompt_data = load_prompt("heat_scorer", "v1.0")
    system_prompt = get_system_prompt("heat_scorer", "v1.0")
    user_template: str = prompt_data.get("user_template", "")

    user_prompt = user_template.format(
        title="AI Model Breaks Benchmark Records Across All Categories",
        source="TechCrunch",
        published_at=datetime.now(timezone.utc).isoformat(),
        snippet=(
            "Researchers announced a breakthrough AI model that surpasses all existing "
            "benchmarks in reasoning, coding, and scientific tasks. Coverage spans "
            "dozens of independent technology publications and has been trending on "
            "social platforms for the past 48 hours. No prior model has achieved this "
            "combination of capabilities simultaneously."
        )[:500],
    )

    _override = os.getenv("MODEL_OVERRIDE", "").strip()
    _model = (_override if _override and not _override.startswith("#") else None) or "gemma4:e4b"

    client = _ollama.AsyncClient(
        host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    response = await client.chat(
        model=_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )
    parsed = json.loads(response["message"]["content"])

    assert "heat_score" in parsed, "heat_score missing from response"
    assert "volume_signal" in parsed, "volume_signal missing from response"
    assert "velocity_signal" in parsed, "velocity_signal missing from response"
    assert "novelty_signal" in parsed, "novelty_signal missing from response"
    assert "significance_signal" in parsed, "significance_signal missing from response"
    assert "reasoning" in parsed, "reasoning missing from response"

    for field in ("heat_score", "volume_signal", "velocity_signal", "novelty_signal", "significance_signal"):
        val = float(parsed[field])
        assert 0.0 <= val <= 1.0, f"{field} = {val} is outside [0.0, 1.0]"


@pytest.mark.asyncio
async def test_live_heat_scorer_json_output_valid(span_tracer, ai_article):
    """Verifies the full pipeline returns parseable structured output."""
    tracer, _ = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-json-01", articles=[ai_article])

    output = await run_agent(inp, tracer=tracer)

    assert output.success is True
    assert len(output.scored_articles) == 1
    scored = output.scored_articles[0]

    # ScoredArticle schema validated by Pydantic — if we got here, JSON was valid
    assert isinstance(scored.heat_score, float)
    assert isinstance(scored.reasoning, str)


# ── OTel span tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_heat_scorer_otel_spans_created(span_tracer, ai_article):
    """Verifies heat_scorer.score and heat_scorer.llm_call spans are emitted."""
    tracer, exporter = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-spans-01", articles=[ai_article])

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("heat_scorer.score" in n for n in span_names), (
        f"Expected heat_scorer.score span. Got: {span_names}"
    )
    assert any("heat_scorer.llm_call" in n for n in span_names), (
        f"Expected heat_scorer.llm_call span. Got: {span_names}"
    )


@pytest.mark.asyncio
async def test_live_heat_scorer_no_content_in_spans(span_tracer, ai_article):
    """Verifies OTel spans contain zero article content — titles, snippets excluded."""
    tracer, exporter = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-content-01", articles=[ai_article])

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()

    _BLOCKED_SPAN_KEYS = {
        "content", "text", "body", "article", "summary",
        "raw", "html", "snippet", "title", "reasoning",
    }
    for span in spans:
        for key in span.attributes:
            assert key.lower() not in _BLOCKED_SPAN_KEYS, (
                f"Span '{span.name}' contains blocked attribute key '{key}'"
            )
        # No attribute value should look like raw content
        for key, value in span.attributes.items():
            if isinstance(value, str):
                assert len(value) < 500, (
                    f"Span '{span.name}' attribute '{key}' may contain raw content "
                    f"(length {len(value)})"
                )


@pytest.mark.asyncio
async def test_live_heat_scorer_score_span_attributes(span_tracer, ai_article):
    """Verifies heat_scorer.score span carries expected structural attributes."""
    tracer, exporter = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-attr-01", articles=[ai_article])

    output = await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    score_spans = [s for s in spans if s.name == "heat_scorer.score"]

    assert score_spans, f"Expected heat_scorer.score span. Got: {[s.name for s in spans]}"
    span = score_spans[0]
    attrs = span.attributes

    assert attrs.get("run_id") == "integration-heat-attr-01"
    assert "model_id" in attrs
    assert "result_count" in attrs
    assert attrs.get("result_count") == 1

    if output.success and output.scored_articles:
        assert "heat_score" in attrs
        assert 0.0 <= attrs["heat_score"] <= 1.0


@pytest.mark.asyncio
async def test_live_heat_scorer_llm_call_span_attributes(span_tracer, ai_article):
    """Verifies heat_scorer.llm_call span carries model_id, prompt_version, token_estimate."""
    tracer, exporter = span_tracer
    inp = HeatScorerInput(run_id="integration-heat-llm-span-01", articles=[ai_article])

    await run_agent(inp, tracer=tracer)

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name == "heat_scorer.llm_call"]

    assert llm_spans, f"Expected heat_scorer.llm_call span. Got: {[s.name for s in spans]}"
    attrs = llm_spans[0].attributes

    assert "model_id" in attrs, "llm_call span missing model_id"
    assert "prompt_version" in attrs, "llm_call span missing prompt_version"
    assert attrs.get("prompt_version") == "v1.0"
    assert "token_estimate" in attrs, "llm_call span missing token_estimate"
    assert isinstance(attrs["token_estimate"], int)
    assert attrs["token_estimate"] > 0


# ── LangSmith tagging test ─────────────────────────────────────────────────────

def test_heat_scorer_langsmith_run_config_tags():
    """Verifies make_run_config produces the required LangSmith tags for filtering."""
    cfg = make_run_config("heat_scorer", "v1.0", run_id="test-run-heat-42")

    assert "agent:heat_scorer" in cfg["tags"], "Missing agent tag"
    assert "prompt:v1.0" in cfg["tags"], "Missing prompt version tag"
    assert "run:test-run-heat-42" in cfg["tags"], "Missing run_id tag"
    assert cfg["metadata"]["prompt_version"] == "v1.0"
    assert cfg["metadata"]["agent_id"] == "heat_scorer"
    assert cfg["run_name"] == "heat_scorer:v1.0"


def test_heat_scorer_traced_run_is_callable():
    """Verifies _traced_run and _call_ollama are callable (LangSmith decorator applied)."""
    from src.agents.heat_scorer.agent import _traced_run, _call_ollama
    assert callable(_traced_run)
    assert callable(_call_ollama)
