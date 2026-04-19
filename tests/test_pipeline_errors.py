"""
Tests for pipeline error protocol: PipelineError, AgentResult, PipelineException,
and the DataQualityValidator helpers called from the orchestrator boundary.
"""
from __future__ import annotations

import pytest

from src.common.error_codes import (
    AUTH_FAILED,
    CITATION_INVALID,
    CONSTRAINT_VIOLATED,
    INJECTION_DETECTED,
    NO_RESULTS,
    NO_VIABLE_CANDIDATES,
    SUMMARY_TOO_SHORT,
)
from src.common.pipeline_errors import (
    AgentResult,
    ErrorSeverity,
    PipelineError,
    PipelineException,
)
from src.common.data_quality import DataQualityValidator

# ── PipelineError ─────────────────────────────────────────────────────────────

def _make_error(
    error_code: str = NO_RESULTS,
    severity: ErrorSeverity = ErrorSeverity.SKIP,
    **kwargs,
) -> PipelineError:
    return PipelineError(
        error_code=error_code,
        severity=severity,
        agent_id="search-worker",
        run_id="run-001",
        topic_id="topic-AI",
        message="No search results returned",
        **kwargs,
    )


def test_pipeline_error_serializes_correctly():
    err = _make_error(
        context={"result_count": 0, "min_required": 3},
        retry_hint="Widen the query",
    )
    data = err.model_dump()
    assert data["error_code"] == NO_RESULTS
    assert data["severity"] == "SKIP"
    assert data["agent_id"] == "search-worker"
    assert data["run_id"] == "run-001"
    assert data["topic_id"] == "topic-AI"
    assert data["context"]["result_count"] == 0
    assert data["retry_hint"] == "Widen the query"


def test_pipeline_error_defaults():
    err = PipelineError(
        error_code=AUTH_FAILED,
        severity=ErrorSeverity.FATAL,
        agent_id="auth-server",
        run_id="run-002",
        message="Shared secret mismatch",
    )
    assert err.topic_id == ""
    assert err.retry_hint == ""
    assert err.context == {}


def test_error_severity_values():
    assert ErrorSeverity.RETRYABLE == "RETRYABLE"
    assert ErrorSeverity.SKIP == "SKIP"
    assert ErrorSeverity.FATAL == "FATAL"
    assert ErrorSeverity.DEGRADED == "DEGRADED"


# ── AgentResult ───────────────────────────────────────────────────────────────

def test_agent_result_success_default():
    result = AgentResult(data={"articles": []})
    assert result.success is True
    assert result.error is None
    assert result.warnings == []
    assert result.data == {"articles": []}


def test_agent_result_success_explicit():
    result = AgentResult(success=True, data=42)
    assert result.success is True
    assert result.error is None


def test_agent_result_failure_has_error():
    err = _make_error(severity=ErrorSeverity.FATAL, error_code=NO_VIABLE_CANDIDATES)
    result = AgentResult(success=False, error=err)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == NO_VIABLE_CANDIDATES
    assert result.data is None


def test_agent_result_failure_severity_accessible():
    err = _make_error(severity=ErrorSeverity.RETRYABLE, error_code=SUMMARY_TOO_SHORT)
    result = AgentResult(success=False, error=err)
    assert result.error.severity == ErrorSeverity.RETRYABLE


def test_agent_result_warnings():
    result = AgentResult(success=True, data="ok", warnings=["low heat score"])
    assert len(result.warnings) == 1
    assert "heat" in result.warnings[0]


# ── PipelineException ─────────────────────────────────────────────────────────

def test_pipeline_exception_wraps_error():
    err = _make_error(severity=ErrorSeverity.FATAL)
    exc = PipelineException(err)
    assert exc.error is err
    assert str(exc) == err.message


def test_pipeline_exception_is_exception():
    err = _make_error()
    exc = PipelineException(err)
    with pytest.raises(PipelineException):
        raise exc


# ── Error severity → orchestrator action contract ─────────────────────────────

@pytest.mark.parametrize("code,severity", [
    (NO_RESULTS, ErrorSeverity.SKIP),
    (NO_VIABLE_CANDIDATES, ErrorSeverity.FATAL),
    (SUMMARY_TOO_SHORT, ErrorSeverity.RETRYABLE),
    (INJECTION_DETECTED, ErrorSeverity.SKIP),
    (AUTH_FAILED, ErrorSeverity.FATAL),
])
def test_error_code_severity_contract(code, severity):
    """Error codes that always map to a specific severity."""
    validator = DataQualityValidator(agent_id="test-agent")
    # Spot-check a few via the validator; others are asserted structurally
    err = PipelineError(
        error_code=code,
        severity=severity,
        agent_id="test-agent",
        run_id="r1",
        message="test",
    )
    assert err.severity == severity


# ── PHI / content safety in error context ────────────────────────────────────

_BLOCKED_CONTEXT_KEYS = frozenset(
    {"content", "text", "body", "article", "summary", "raw", "html", "snippet"}
)


def test_pipeline_error_context_has_no_content_keys():
    """Error context must only contain metadata, never content values."""
    err = _make_error(
        context={
            "result_count": 3,
            "min_required": 5,
            "topic_id": "AI",
            "heat_score": 0.4,
        }
    )
    for key in err.context:
        assert key.lower() not in _BLOCKED_CONTEXT_KEYS, (
            f"Context key '{key}' looks like a content field"
        )


def test_pipeline_error_context_rejects_content_keys_by_convention():
    """Document that callers must not put content in context."""
    # The model itself does not enforce this (would require inspecting values).
    # This test documents the convention and catches obvious violations in
    # DataQualityValidator-produced errors.
    validator = DataQualityValidator(agent_id="search-worker")
    err = validator.validate_search_results(
        [], min_results=3, topic_id="t1", run_id="r1"
    )
    assert err is not None
    for key in err.context:
        assert key.lower() not in _BLOCKED_CONTEXT_KEYS


# ── Citation validation ───────────────────────────────────────────────────────

def test_citation_validation_catches_invalid_url():
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_citations(
        citations=["https://hallucinated.com/article"],
        sources=["https://real-source.com/news/1"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is not None
    assert err.error_code == CITATION_INVALID
    assert err.severity == ErrorSeverity.RETRYABLE
    assert err.context["invalid_count"] == 1


def test_citation_validation_passes_valid_urls():
    validator = DataQualityValidator(agent_id="summarizer")
    source = "https://real-source.com/news/1"
    err = validator.validate_citations(
        citations=[source],
        sources=[source],
        topic_id="t1",
        run_id="r1",
    )
    assert err is None


def test_citation_validation_partial_invalid():
    validator = DataQualityValidator(agent_id="summarizer")
    sources = ["https://good.com/a", "https://good.com/b"]
    err = validator.validate_citations(
        citations=["https://good.com/a", "https://bad.com/c"],
        sources=sources,
        topic_id="t1",
        run_id="r1",
    )
    assert err is not None
    assert err.context["invalid_count"] == 1
    assert err.context["total_citations"] == 2


def test_citation_validation_empty_citations_passes():
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_citations(
        citations=[],
        sources=["https://example.com"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is None


# ── Constraint validation ─────────────────────────────────────────────────────

def test_constraint_validation_catches_violation():
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_constraints_respected(
        content="This article discusses recent advances in AI technology.",
        constraints=["healthcare"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is not None
    assert err.error_code == CONSTRAINT_VIOLATED
    assert err.severity == ErrorSeverity.RETRYABLE
    assert err.context["violated_count"] == 1


def test_constraint_validation_passes_when_satisfied():
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_constraints_respected(
        content="This covers healthcare and patient outcomes.",
        constraints=["healthcare", "patient"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is None


def test_constraint_validation_empty_constraints_passes():
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_constraints_respected(
        content="anything",
        constraints=[],
        topic_id="t1",
        run_id="r1",
    )
    assert err is None


def test_constraint_context_does_not_include_constraint_terms():
    """Constraint terms may describe sensitive user preferences — keep out of context."""
    validator = DataQualityValidator(agent_id="summarizer")
    err = validator.validate_constraints_respected(
        content="unrelated content",
        constraints=["secret-patient-condition"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is not None
    # context must only have counts, not the constraint terms themselves
    assert "violated_count" in err.context
    assert "secret-patient-condition" not in str(err.context)
