"""
Pydantic request/response schemas for the News MAS gateway API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Topics ────────────────────────────────────────────────────────────────────

class TopicCreate(BaseModel):
    topic_text: str
    constraints: list[str] = Field(default_factory=list)


class TopicUpdate(BaseModel):
    topic_text: Optional[str] = None
    constraints: Optional[list[str]] = None
    active: Optional[bool] = None


class TopicResponse(BaseModel):
    topic_id: str
    user_id: str
    topic_text: str
    constraints: list[str]
    active: bool
    created_at: datetime
    updated_at: datetime
    last_run_id: Optional[str] = None
    last_heat_score: Optional[float] = None


# ── Digests ───────────────────────────────────────────────────────────────────

class DigestSummary(BaseModel):
    run_id: str
    generated_at: Optional[datetime] = None
    passed_count: int = 0
    tombstoned_count: int = 0
    overall_confidence: str = "LOW"
    run_status: str = "unknown"


# ── Runs ──────────────────────────────────────────────────────────────────────

class RunTriggerResponse(BaseModel):
    run_id: str
    status: str = "triggered"


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: str = "api"
    error: Optional[str] = None


class RunHistoryEntry(BaseModel):
    run_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    passed_count: int = 0
    tombstoned_count: int = 0
    triggered_by: str = "api"


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackSummaryRequest(BaseModel):
    run_id: str
    article_url: str
    vote: Literal["up", "down"]
    note: Optional[str] = None


class FeedbackTopicRequest(BaseModel):
    topic_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    note: Optional[str] = None


class FeedbackMissedRequest(BaseModel):
    topic_text: str
    note: Optional[str] = None


class FeedbackOverrideRequest(BaseModel):
    run_id: str
    topic_id: str
    action: Literal["add", "remove"]
    note: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "accepted"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "news-mas-gateway"
    timestamp: datetime
