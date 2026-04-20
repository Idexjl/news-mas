from datetime import datetime

import pytest

from src.common.schemas import (
    DigestEntry,
    FilterAgentInput,
    FilterAgentOutput,
    HeatScorerInput,
    HeatScorerOutput,
    JudgedArticle,
    Phase1JudgeInput,
    Phase1JudgeOutput,
    RawArticle,
    RelevanceGateInput,
    RelevanceGateOutput,
    ReviewerInput,
    ReviewerOutput,
    RunState,
    RunStatus,
    ScoredArticle,
    SearchResult,
    SearchWorkerInput,
    SearchWorkerOutput,
    SelectorInput,
    SelectorOutput,
    SummarizerInput,
    SummarizerOutput,
)


@pytest.fixture
def article():
    return SearchResult(
        url="https://example.com/news/1",
        title="Test Article",
        content="Body text here.",
        published_at=datetime(2026, 4, 18, 12, 0, 0),
        source="example.com",
    )


@pytest.fixture
def scored(article):
    return ScoredArticle(article=article, heat_score=0.75, reasoning="High impact story")


# ── SearchWorker ──────────────────────────────────────────────────────────────

def test_search_worker_input_defaults():
    inp = SearchWorkerInput(run_id="r1", topics=["AI", "tech"])
    assert inp.since_days == 7
    assert inp.max_results_per_topic == 10


def test_search_worker_output(article):
    out = SearchWorkerOutput(run_id="r1", articles=[article])
    assert len(out.articles) == 1
    assert out.articles[0].url == "https://example.com/news/1"


# ── HeatScorer ────────────────────────────────────────────────────────────────

def test_heat_scorer_input(article):
    inp = HeatScorerInput(run_id="r1", articles=[article])
    assert len(inp.articles) == 1


def test_heat_scorer_output_score_bounds(scored):
    out = HeatScorerOutput(run_id="r1", scored_articles=[scored])
    assert 0.0 <= out.scored_articles[0].heat_score <= 1.0


def test_heat_scorer_score_validation():
    with pytest.raises(Exception):
        ScoredArticle(
            article=RawArticle(url="x", title="t", content="c"),
            heat_score=1.5,  # out of range
        )


# ── FilterAgent ───────────────────────────────────────────────────────────────

def test_filter_agent_io(scored):
    inp = FilterAgentInput(run_id="r1", scored_articles=[scored])
    out = FilterAgentOutput(run_id="r1", filtered_articles=[scored], dropped_count=3)
    assert out.dropped_count == 3
    assert len(out.filtered_articles) == 1


# ── Selector ──────────────────────────────────────────────────────────────────

def test_selector_io(scored):
    inp = SelectorInput(run_id="r1", filtered_articles=[scored], max_select=5)
    out = SelectorOutput(run_id="r1", selected_articles=[scored])
    assert len(out.selected_articles) == 1


# ── Phase1Judge ───────────────────────────────────────────────────────────────

def test_phase1_judge_io(article, scored):
    inp = Phase1JudgeInput(run_id="r1", selected_articles=[scored])
    judged = JudgedArticle(article=article, approved=True, reason="On topic")
    out = Phase1JudgeOutput(run_id="r1", judged_articles=[judged])
    assert out.judged_articles[0].approved is True


# ── Summarizer ────────────────────────────────────────────────────────────────

def test_summarizer_io(article):
    inp = SummarizerInput(run_id="r1", article=article)
    out = SummarizerOutput(run_id="r1", summary="Short summary.", key_points=["Point A"])
    assert out.key_points == ["Point A"]


# ── Reviewer ──────────────────────────────────────────────────────────────────

def test_reviewer_approved(article):
    inp = ReviewerInput(run_id="r1", article=article, summary="S", key_points=[])
    out = ReviewerOutput(run_id="r1", approved=True, feedback="Looks good")
    assert out.revised_summary is None


def test_reviewer_with_revision(article):
    out = ReviewerOutput(
        run_id="r1", approved=False, revised_summary="Better summary", feedback="Reworded"
    )
    assert out.revised_summary == "Better summary"


# ── RelevanceGate ─────────────────────────────────────────────────────────────

def test_relevance_gate_io(article):
    inp = RelevanceGateInput(run_id="r1", article=article, summary="S")
    out = RelevanceGateOutput(run_id="r1", is_relevant=True, confidence=0.92, reason="On topic")
    assert 0.0 <= out.confidence <= 1.0


# ── DigestEntry ───────────────────────────────────────────────────────────────

def test_digest_entry(article):
    entry = DigestEntry(
        article=article,
        summary="Final summary.",
        key_points=["Key 1", "Key 2"],
        heat_score=0.8,
        relevance_confidence=0.95,
    )
    assert entry.relevance_confidence == 0.95


# ── RunState ──────────────────────────────────────────────────────────────────

def test_run_state_defaults():
    state = RunState(run_id="r1", topics=["AI"])
    assert state.status == RunStatus.PENDING
    assert state.digest == []
    assert state.error is None
    assert isinstance(state.started_at, datetime)


def test_run_status_values():
    assert RunStatus.PENDING == "pending"
    assert RunStatus.RUNNING == "running"
    assert RunStatus.COMPLETED == "completed"
    assert RunStatus.FAILED == "failed"
    assert RunStatus.PARTIAL == "partial"
