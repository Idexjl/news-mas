from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.common.pipeline_errors import AgentResult, ResultConfidence  # noqa: F401 — re-exported for agent use


# ── Shared primitives ─────────────────────────────────────────────────────────

class RawArticle(BaseModel):
    url: str
    title: str
    content: str
    published_at: Optional[datetime] = None
    source: str = ""


class SearchResult(RawArticle):
    """RawArticle annotated with source-quality confidence set by SearchWorker."""
    confidence: ResultConfidence = ResultConfidence.FULL
    confidence_reason: Optional[str] = None


class CandidateConfidence(BaseModel):
    """Aggregated source-quality metrics for a batch of search results."""
    topic_id: str = ""
    heat_score: float = 0.0
    total_results: int = 0
    full_content_count: int = 0
    snippet_only_count: int = 0
    partial_count: int = 0
    injection_detected_count: int = 0
    phi_scrubbed_count: int = 0
    confidence_ratio: float = 0.0
    overall_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"

    @classmethod
    def from_results(
        cls,
        results: list[SearchResult],
        *,
        topic_id: str = "",
        heat_score: float = 0.0,
        high_threshold: float | None = None,
        medium_threshold: float | None = None,
    ) -> "CandidateConfidence":
        _high = high_threshold if high_threshold is not None else float(
            os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.7")
        )
        _medium = medium_threshold if medium_threshold is not None else float(
            os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.3")
        )
        total = len(results)
        if total == 0:
            return cls(topic_id=topic_id, heat_score=heat_score)

        full = sum(1 for r in results if r.confidence == ResultConfidence.FULL)
        snippet = sum(1 for r in results if r.confidence == ResultConfidence.SNIPPET)
        partial = sum(1 for r in results if r.confidence == ResultConfidence.PARTIAL)
        injected = sum(1 for r in results if r.confidence == ResultConfidence.INJECTED)
        scrubbed = sum(1 for r in results if r.confidence == ResultConfidence.SCRUBBED)

        ratio = full / total

        if injected > 0:
            overall: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
        elif ratio >= _high:
            overall = "HIGH"
        elif ratio >= _medium:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        return cls(
            topic_id=topic_id,
            heat_score=heat_score,
            total_results=total,
            full_content_count=full,
            snippet_only_count=snippet,
            partial_count=partial,
            injection_detected_count=injected,
            phi_scrubbed_count=scrubbed,
            confidence_ratio=ratio,
            overall_confidence=overall,
        )


class ScoredArticle(BaseModel):
    article: RawArticle
    heat_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class JudgedArticle(BaseModel):
    article: RawArticle
    approved: bool
    reason: str = ""


# ── Search Worker ─────────────────────────────────────────────────────────────

class SearchWorkerInput(BaseModel):
    run_id: str
    topics: list[str]
    since_days: int = Field(default=7, ge=1, le=90)
    max_results_per_topic: int = Field(default=10, ge=1, le=50)
    aap_token: Optional[str] = None  # validated when present; omit in dev/test


class SearchWorkerOutput(AgentResult):
    run_id: str
    articles: list[SearchResult] = Field(default_factory=list)


# ── Heat Scorer ───────────────────────────────────────────────────────────────

class HeatScorerInput(BaseModel):
    run_id: str
    articles: list[SearchResult]
    aap_token: Optional[str] = None  # validated when present; omit in dev/test


class HeatScorerOutput(AgentResult):
    run_id: str
    scored_articles: list[ScoredArticle] = Field(default_factory=list)
    candidate_confidence: Optional[CandidateConfidence] = None


# ── Filter Agent ──────────────────────────────────────────────────────────────

class RemovalReason(BaseModel):
    """Records why a result was removed. Constraint text is never stored here."""
    url: str
    constraint_violated: int  # index into constraints list; -1 = auto-removed (injected)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str = ""  # e.g. "snippet only — limited context"; never contains constraint text


class FilterAgentInput(BaseModel):
    run_id: str
    scored_articles: list[ScoredArticle]
    min_heat_score: float = Field(default=0.5, ge=0.0, le=1.0)
    constraints: list[str] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    aap_token: Optional[str] = None


class FilterAgentOutput(AgentResult):
    run_id: str
    filtered_articles: list[ScoredArticle] = Field(default_factory=list)
    dropped_count: int = 0
    kept_results: list[ScoredArticle] = Field(default_factory=list)
    removed_results: list[ScoredArticle] = Field(default_factory=list)
    removal_reasons: list[RemovalReason] = Field(default_factory=list)


# ── Selector ──────────────────────────────────────────────────────────────────

class ScoredFilteredTopic(BaseModel):
    """Per-topic bundle passed to the Selector after scoring and filtering."""
    topic_id: str
    topic_text: str  # user-defined query — never logged or recorded in spans
    heat_score: float = Field(default=0.0, ge=0.0, le=1.0)
    kept_results: list[ScoredArticle] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    removal_reasons: list[RemovalReason] = Field(default_factory=list)


class SelectedTopic(BaseModel):
    topic_id: str
    rank: int
    selection_reasoning: str
    confidence_assessment: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"


class RejectedTopic(BaseModel):
    topic_id: str
    reject_reason: str


class SelectorInput(BaseModel):
    run_id: str
    topics: list[ScoredFilteredTopic]
    max_select: int = Field(default=5, ge=1, le=50)
    topic_memory_context: Optional[dict] = None  # {topic_id: [run_outcome, ...]}
    aap_token: Optional[str] = None


class SelectorOutput(AgentResult):
    run_id: str
    selected: list[SelectedTopic] = Field(default_factory=list)
    rejected: list[RejectedTopic] = Field(default_factory=list)
    selection_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    diversity_note: Optional[str] = None
    selected_articles: list[ScoredArticle] = Field(default_factory=list)


# ── Phase 1 Judge ─────────────────────────────────────────────────────────────

class JudgeAdjustment(BaseModel):
    """A single swap the Phase 1 Judge made to the selector's choices."""
    action: Literal["swap"] = "swap"
    removed_topic_id: str
    added_topic_id: str
    reason_code: Literal["confidence_upgrade", "diversity_improvement", "quality_concern"]


class JudgedTopic(BaseModel):
    """A topic in the judge's final selection, annotated with provenance."""
    topic_id: str
    rank: int
    source: Literal["selector", "judge"] = "selector"
    judge_reasoning: Optional[str] = None  # set only when source == "judge"


class Phase1JudgeInput(BaseModel):
    run_id: str
    # Full-picture topic-level inputs from the selector
    all_candidates: list[ScoredFilteredTopic] = Field(default_factory=list)
    selected: list[SelectedTopic] = Field(default_factory=list)
    rejected: list[RejectedTopic] = Field(default_factory=list)
    candidate_confidence: Optional[CandidateConfidence] = None
    since_date: Optional[str] = None
    aap_token: Optional[str] = None
    # Retained for Phase 2 compatibility (flat article list from selector output)
    selected_articles: list[ScoredArticle] = Field(default_factory=list)


class Phase1JudgeOutput(AgentResult):
    run_id: str
    verdict: Literal["endorsed", "adjusted"] = "endorsed"
    final_selected: list[JudgedTopic] = Field(default_factory=list)
    adjustments: list[JudgeAdjustment] = Field(default_factory=list)
    endorsement_note: Optional[str] = None
    overall_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    # Flat approved-article list consumed by Phase 2 orchestrator
    judged_articles: list[JudgedArticle] = Field(default_factory=list)


# ── Summarizer ────────────────────────────────────────────────────────────────

class SummarizerInput(BaseModel):
    run_id: str
    article: RawArticle


class SummarizerOutput(AgentResult):
    run_id: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)


# ── Reviewer ──────────────────────────────────────────────────────────────────

class ReviewerInput(BaseModel):
    run_id: str
    article: RawArticle
    summary: str
    key_points: list[str]


class ReviewerOutput(AgentResult):
    run_id: str
    approved: bool = False
    revised_summary: Optional[str] = None
    feedback: str = ""


# ── Relevance Gate ────────────────────────────────────────────────────────────

class RelevanceGateInput(BaseModel):
    run_id: str
    article: RawArticle
    summary: str


class RelevanceGateOutput(AgentResult):
    run_id: str
    is_relevant: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


# ── Digest / Run State ────────────────────────────────────────────────────────

class DigestEntry(BaseModel):
    article: RawArticle
    summary: str
    key_points: list[str]
    heat_score: float
    relevance_confidence: float
    confidence: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = None
    confidence_note: Optional[str] = None


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RunState(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    topics: list[str]
    digest: list[DigestEntry] = Field(default_factory=list)
    error: Optional[str] = None
    metrics: dict = Field(default_factory=dict)


# ── Agent registry ────────────────────────────────────────────────────────────

class WorkloadIdentityInfo(BaseModel):
    """Workload identity state for one agent as reported in the registry."""
    provider_type: str = Field(
        description=(
            "Active workload identity provider. "
            "One of: local | azure-managed-identity | aws-sts | gcp-workload"
        )
    )
    verified: bool = Field(
        default=False,
        description=(
            "True once the provider has successfully obtained and validated "
            "a platform-issued identity token at startup."
        ),
    )


class AgentCard(BaseModel):
    """Registry entry describing one agent's static and runtime metadata."""
    agent_name: str
    port: int
    phase: int = Field(ge=1, le=2)
    model: Optional[str] = None
    workload_identity: WorkloadIdentityInfo
