"""
Tests for result confidence taxonomy:
  - ResultConfidence enum values
  - SearchResult confidence field assignment
  - CandidateConfidence aggregation (HIGH/MEDIUM/LOW thresholds)
  - injection_detected forces LOW regardless of ratio
  - DigestEntry carries confidence field
"""
from __future__ import annotations

import pytest

from src.common.pipeline_errors import ResultConfidence
from src.common.schemas import (
    CandidateConfidence,
    DigestEntry,
    RawArticle,
    SearchResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(confidence: ResultConfidence, url: str = "https://example.com/a") -> SearchResult:
    return SearchResult(
        url=url,
        title="Test Article",
        content="Some content.",
        source="example.com",
        confidence=confidence,
    )


def _make_raw_article(url: str = "https://example.com/a") -> RawArticle:
    return RawArticle(url=url, title="T", content="C", source="s")


# ── ResultConfidence enum ─────────────────────────────────────────────────────

def test_result_confidence_values():
    assert ResultConfidence.FULL == "full"
    assert ResultConfidence.SNIPPET == "snippet"
    assert ResultConfidence.PARTIAL == "partial"
    assert ResultConfidence.INJECTED == "injected"
    assert ResultConfidence.SCRUBBED == "scrubbed"


# ── SearchResult ──────────────────────────────────────────────────────────────

def test_search_result_full_confidence():
    r = _make_result(ResultConfidence.FULL)
    assert r.confidence == ResultConfidence.FULL
    assert r.confidence_reason is None


def test_search_result_snippet_confidence():
    r = SearchResult(
        url="https://example.com/b",
        title="T",
        content="snippet only",
        source="s",
        confidence=ResultConfidence.SNIPPET,
        confidence_reason="fetch_failed",
    )
    assert r.confidence == ResultConfidence.SNIPPET
    assert r.confidence_reason == "fetch_failed"


def test_search_result_partial_confidence():
    r = _make_result(ResultConfidence.PARTIAL)
    r2 = SearchResult(**{**r.model_dump(), "confidence_reason": "truncated_at_token_budget"})
    assert r2.confidence == ResultConfidence.PARTIAL
    assert "truncated" in r2.confidence_reason


def test_search_result_injected_confidence():
    r = SearchResult(
        url="https://evil.com/inject",
        title="",
        content="",
        source="evil.com",
        confidence=ResultConfidence.INJECTED,
        confidence_reason="injection_patterns:3",
    )
    assert r.confidence == ResultConfidence.INJECTED
    assert r.content == ""


def test_search_result_scrubbed_confidence():
    r = SearchResult(
        url="https://example.com/phi",
        title="Dr <PERSON>",
        content="Patient <US_SSN> discharged.",
        source="hospital.com",
        confidence=ResultConfidence.SCRUBBED,
        confidence_reason="phi_entities:2",
    )
    assert r.confidence == ResultConfidence.SCRUBBED


def test_search_result_default_confidence_is_full():
    r = SearchResult(url="https://x.com", title="T", content="C", source="s")
    assert r.confidence == ResultConfidence.FULL


def test_raw_article_coerces_to_search_result():
    """RawArticle passed to a SearchResult-typed field gets default FULL confidence."""
    raw = _make_raw_article()
    sr = SearchResult.model_validate(raw.model_dump())
    assert sr.url == raw.url
    assert sr.confidence == ResultConfidence.FULL


# ── CandidateConfidence ───────────────────────────────────────────────────────

def test_candidate_confidence_high():
    results = [_make_result(ResultConfidence.FULL, f"https://x.com/{i}") for i in range(8)]
    results += [_make_result(ResultConfidence.SNIPPET, f"https://y.com/{i}") for i in range(2)]
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.total_results == 10
    assert cc.full_content_count == 8
    assert cc.snippet_only_count == 2
    assert cc.confidence_ratio == pytest.approx(0.8)
    assert cc.overall_confidence == "HIGH"


def test_candidate_confidence_medium():
    results = [_make_result(ResultConfidence.FULL, f"https://x.com/{i}") for i in range(4)]
    results += [_make_result(ResultConfidence.SNIPPET, f"https://y.com/{i}") for i in range(6)]
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.confidence_ratio == pytest.approx(0.4)
    assert cc.overall_confidence == "MEDIUM"


def test_candidate_confidence_low():
    # SCRUBBED articles count toward total but contribute nothing to ratio → LOW
    results = [_make_result(ResultConfidence.SCRUBBED, f"https://y.com/{i}") for i in range(10)]
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.confidence_ratio == pytest.approx(0.0)
    assert cc.overall_confidence == "LOW"


def test_candidate_confidence_snippet_floor_at_least_medium():
    # Snippet-only results (zero successful fetches) are floored to MEDIUM.
    results = [_make_result(ResultConfidence.SNIPPET, f"https://y.com/{i}") for i in range(10)]
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.confidence_ratio == pytest.approx(0.0)
    assert cc.snippet_only_count == 10
    assert cc.overall_confidence == "MEDIUM"


def test_candidate_confidence_snippet_floor_blocked_by_injection():
    # Injection override takes priority over snippet floor — result is still LOW.
    results = [_make_result(ResultConfidence.SNIPPET, f"https://y.com/{i}") for i in range(9)]
    results.append(_make_result(ResultConfidence.INJECTED, "https://evil.com/0"))
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.injection_detected_count == 1
    assert cc.overall_confidence == "LOW"


def test_injection_detected_forces_low_regardless_of_ratio():
    # Even with 9/10 FULL results, one INJECTED forces LOW.
    results = [_make_result(ResultConfidence.FULL, f"https://x.com/{i}") for i in range(9)]
    results.append(_make_result(ResultConfidence.INJECTED, "https://evil.com/0"))
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.injection_detected_count == 1
    assert cc.overall_confidence == "LOW"
    assert cc.confidence_ratio == pytest.approx(0.9)  # ratio is high but overall is still LOW


def test_candidate_confidence_counts_all_types():
    results = [
        _make_result(ResultConfidence.FULL, "https://x.com/full"),
        _make_result(ResultConfidence.SNIPPET, "https://x.com/snip"),
        _make_result(ResultConfidence.PARTIAL, "https://x.com/part"),
        _make_result(ResultConfidence.INJECTED, "https://x.com/inj"),
        _make_result(ResultConfidence.SCRUBBED, "https://x.com/scrub"),
    ]
    cc = CandidateConfidence.from_results(results, high_threshold=0.7, medium_threshold=0.3)
    assert cc.total_results == 5
    assert cc.full_content_count == 1
    assert cc.snippet_only_count == 1
    assert cc.partial_count == 1
    assert cc.injection_detected_count == 1
    assert cc.phi_scrubbed_count == 1


def test_candidate_confidence_empty_results():
    cc = CandidateConfidence.from_results([], high_threshold=0.7, medium_threshold=0.3)
    assert cc.total_results == 0
    assert cc.confidence_ratio == 0.0
    assert cc.overall_confidence == "LOW"


def test_candidate_confidence_serializes_correctly():
    results = [_make_result(ResultConfidence.FULL, "https://x.com/a")]
    cc = CandidateConfidence.from_results(
        results, topic_id="AI", heat_score=0.75, high_threshold=0.7, medium_threshold=0.3
    )
    data = cc.model_dump()
    assert data["topic_id"] == "AI"
    assert data["heat_score"] == pytest.approx(0.75)
    assert data["overall_confidence"] == "HIGH"
    assert data["full_content_count"] == 1
    assert data["confidence_ratio"] == pytest.approx(1.0)


# ── DigestEntry carries confidence ────────────────────────────────────────────

def test_digest_entry_carries_confidence():
    article = _make_raw_article()
    entry = DigestEntry(
        article=article,
        summary="Summary text.",
        key_points=["Point A"],
        heat_score=0.8,
        relevance_confidence=0.9,
        confidence="HIGH",
        confidence_note="Strong source coverage",
    )
    assert entry.confidence == "HIGH"
    assert entry.confidence_note == "Strong source coverage"


def test_digest_entry_confidence_medium_note():
    article = _make_raw_article()
    entry = DigestEntry(
        article=article,
        summary="Summary.",
        key_points=[],
        heat_score=0.5,
        relevance_confidence=0.6,
        confidence="MEDIUM",
        confidence_note="Based on limited source content",
    )
    assert entry.confidence == "MEDIUM"


def test_digest_entry_confidence_low_safety_note():
    article = _make_raw_article()
    entry = DigestEntry(
        article=article,
        summary="Summary.",
        key_points=[],
        heat_score=0.4,
        relevance_confidence=0.5,
        confidence="LOW",
        confidence_note="Sources had content removed for safety",
    )
    assert entry.confidence == "LOW"


def test_digest_entry_confidence_optional_defaults_none():
    """Existing code that does not set confidence still works."""
    article = _make_raw_article()
    entry = DigestEntry(
        article=article,
        summary="S.",
        key_points=[],
        heat_score=0.7,
        relevance_confidence=0.8,
    )
    assert entry.confidence is None
    assert entry.confidence_note is None
