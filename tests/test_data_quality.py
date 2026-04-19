"""
Tests for DataQualityValidator covering the six required error scenarios.
Each test verifies the specific error code, severity, and context shape.
"""
from __future__ import annotations

import pytest

from src.common.data_quality import DataQualityValidator
from src.common.error_codes import (
    ALL_FILTERED,
    CITATION_INVALID,
    HEAT_SCORE_TOO_LOW,
    INSUFFICIENT_RESULTS,
    NO_RESULTS,
    SUMMARY_TOO_SHORT,
)
from src.common.pipeline_errors import ErrorSeverity


@pytest.fixture
def v() -> DataQualityValidator:
    return DataQualityValidator(agent_id="test-agent")


# ── Search results ────────────────────────────────────────────────────────────

def test_empty_results_produces_no_results_skip(v):
    err = v.validate_search_results([], min_results=3, topic_id="t1", run_id="r1")
    assert err is not None
    assert err.error_code == NO_RESULTS
    assert err.severity == ErrorSeverity.SKIP
    assert err.context["result_count"] == 0


def test_single_result_below_min_produces_insufficient_degraded(v):
    err = v.validate_search_results(
        [object()], min_results=3, topic_id="t1", run_id="r1"
    )
    assert err is not None
    assert err.error_code == INSUFFICIENT_RESULTS
    assert err.severity == ErrorSeverity.DEGRADED
    assert err.context["result_count"] == 1
    assert err.context["min_required"] == 3


def test_results_at_min_threshold_passes(v):
    results = [object(), object(), object()]
    err = v.validate_search_results(results, min_results=3, topic_id="t1", run_id="r1")
    assert err is None


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_all_results_filtered_produces_all_filtered_skip(v):
    err = v.validate_filtered_results(
        filtered_count=0, original_count=5, run_id="r1", topic_id="t1"
    )
    assert err is not None
    assert err.error_code == ALL_FILTERED
    assert err.severity == ErrorSeverity.SKIP
    assert err.context["original_count"] == 5
    assert err.context["filtered_count"] == 0


def test_partial_filtering_passes(v):
    err = v.validate_filtered_results(
        filtered_count=2, original_count=5, run_id="r1", topic_id="t1"
    )
    assert err is None


def test_empty_original_does_not_produce_all_filtered(v):
    err = v.validate_filtered_results(
        filtered_count=0, original_count=0, run_id="r1", topic_id="t1"
    )
    assert err is None


# ── Heat score ────────────────────────────────────────────────────────────────

def test_heat_score_zero_produces_heat_score_too_low_skip(v):
    err = v.validate_heat_score(0.0, topic_id="t1", run_id="r1")
    assert err is not None
    assert err.error_code == HEAT_SCORE_TOO_LOW
    assert err.severity == ErrorSeverity.SKIP
    assert err.context["heat_score"] == 0.0


def test_negative_heat_score_produces_heat_score_too_low(v):
    err = v.validate_heat_score(-0.1, topic_id="t1", run_id="r1")
    assert err is not None
    assert err.error_code == HEAT_SCORE_TOO_LOW


def test_positive_heat_score_passes(v):
    err = v.validate_heat_score(0.01, topic_id="t1", run_id="r1")
    assert err is None


# ── Summary length ────────────────────────────────────────────────────────────

def test_short_summary_produces_summary_too_short_retryable(v):
    err = v.validate_summary("Hi", min_length=50, max_length=500, topic_id="t1", run_id="r1")
    assert err is not None
    assert err.error_code == SUMMARY_TOO_SHORT
    assert err.severity == ErrorSeverity.RETRYABLE
    assert err.context["summary_length"] == 2
    assert err.context["min_length"] == 50


def test_valid_summary_passes(v):
    summary = "x" * 100
    err = v.validate_summary(summary, min_length=50, max_length=500, topic_id="t1", run_id="r1")
    assert err is None


# ── Citation validation ───────────────────────────────────────────────────────

def test_invalid_citation_url_produces_citation_invalid_retryable(v):
    err = v.validate_citations(
        citations=["https://made-up.example.com/fake"],
        sources=["https://real.example.com/article"],
        topic_id="t1",
        run_id="r1",
    )
    assert err is not None
    assert err.error_code == CITATION_INVALID
    assert err.severity == ErrorSeverity.RETRYABLE
    assert err.context["invalid_count"] == 1
