from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Shared primitives ─────────────────────────────────────────────────────────

class RawArticle(BaseModel):
    url: str
    title: str
    content: str
    published_at: Optional[datetime] = None
    source: str = ""


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


class SearchWorkerOutput(BaseModel):
    run_id: str
    articles: list[RawArticle]


# ── Heat Scorer ───────────────────────────────────────────────────────────────

class HeatScorerInput(BaseModel):
    run_id: str
    articles: list[RawArticle]


class HeatScorerOutput(BaseModel):
    run_id: str
    scored_articles: list[ScoredArticle]


# ── Filter Agent ──────────────────────────────────────────────────────────────

class FilterAgentInput(BaseModel):
    run_id: str
    scored_articles: list[ScoredArticle]
    min_heat_score: float = Field(default=0.5, ge=0.0, le=1.0)


class FilterAgentOutput(BaseModel):
    run_id: str
    filtered_articles: list[ScoredArticle]
    dropped_count: int = 0


# ── Selector ──────────────────────────────────────────────────────────────────

class SelectorInput(BaseModel):
    run_id: str
    filtered_articles: list[ScoredArticle]
    max_select: int = Field(default=10, ge=1, le=50)


class SelectorOutput(BaseModel):
    run_id: str
    selected_articles: list[ScoredArticle]


# ── Phase 1 Judge ─────────────────────────────────────────────────────────────

class Phase1JudgeInput(BaseModel):
    run_id: str
    selected_articles: list[ScoredArticle]


class Phase1JudgeOutput(BaseModel):
    run_id: str
    judged_articles: list[JudgedArticle]


# ── Summarizer ────────────────────────────────────────────────────────────────

class SummarizerInput(BaseModel):
    run_id: str
    article: RawArticle


class SummarizerOutput(BaseModel):
    run_id: str
    summary: str
    key_points: list[str]


# ── Reviewer ──────────────────────────────────────────────────────────────────

class ReviewerInput(BaseModel):
    run_id: str
    article: RawArticle
    summary: str
    key_points: list[str]


class ReviewerOutput(BaseModel):
    run_id: str
    approved: bool
    revised_summary: Optional[str] = None
    feedback: str = ""


# ── Relevance Gate ────────────────────────────────────────────────────────────

class RelevanceGateInput(BaseModel):
    run_id: str
    article: RawArticle
    summary: str


class RelevanceGateOutput(BaseModel):
    run_id: str
    is_relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


# ── Digest / Run State ────────────────────────────────────────────────────────

class DigestEntry(BaseModel):
    article: RawArticle
    summary: str
    key_points: list[str]
    heat_score: float
    relevance_confidence: float


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
