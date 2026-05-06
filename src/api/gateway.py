"""
FastAPI gateway for the news-mas React frontend.

Exposes REST endpoints for topic management, digest retrieval,
pipeline triggering, feedback capture, and health checks.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.schemas import (
    DigestSummary,
    FeedbackMissedRequest,
    FeedbackOverrideRequest,
    FeedbackResponse,
    FeedbackSummaryRequest,
    FeedbackTopicRequest,
    HealthResponse,
    RunHistoryEntry,
    RunStatusResponse,
    RunTriggerResponse,
    TopicCreate,
    TopicResponse,
    TopicUpdate,
)
from src.common.observability import get_logger, get_tracer, setup_telemetry
from src.common.schemas import TopicInput
from src.common.storage import (
    DigestRepository,
    FeedbackRepository,
    FernetStorage,
    RunRepository,
)
from src.common.topic_repository import TopicRepository, TopicSubscription
from src.orchestrators.pipeline import run_full_pipeline

USER_ID = "user-joe"
SERVICE_NAME = "news-mas-gateway"

setup_telemetry(SERVICE_NAME)
logger = get_logger(__name__)
tracer = get_tracer(SERVICE_NAME)

# In-memory run state for gateway-triggered runs.
# Keys are run_ids; values are dicts with status, timestamps, and counts.
_run_status: dict[str, dict[str, Any]] = {}


# ── Middleware ─────────────────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = FernetStorage()
    app.state.topic_repo = TopicRepository(storage)
    app.state.digest_repo = DigestRepository(storage)
    app.state.feedback_repo = FeedbackRepository(storage)
    app.state.run_repo = RunRepository(storage)
    logger.info("gateway.startup")
    yield
    logger.info("gateway.shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="News MAS Gateway", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_pipeline_task(run_id: str, topic_repo: TopicRepository) -> None:
    """Fire-and-forget: runs the full pipeline and updates _run_status."""
    try:
        topics = topic_repo.get_all_topics()
    except Exception as exc:
        _run_status[run_id].update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": type(exc).__name__,
        })
        logger.error("gateway.pipeline_topics_failed", extra={
            "run_id": run_id, "error_type": type(exc).__name__,
        })
        return

    _run_status[run_id]["status"] = "running"

    topic_inputs = [
        TopicInput(
            topic_id=t.topic_id,
            topic_text=t.topic_text,
            constraints=t.constraints,
        )
        for t in topics
    ]
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()

    try:
        result = await run_full_pipeline(topic_inputs, since_date, USER_ID, run_id=run_id)
        _run_status[run_id].update({
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "passed_count": result.passed_count,
            "tombstoned_count": result.tombstoned_count,
        })
        logger.info("gateway.pipeline_complete", extra={
            "run_id": run_id,
            "passed_count": result.passed_count,
            "tombstoned_count": result.tombstoned_count,
        })
    except Exception as exc:
        _run_status[run_id].update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": type(exc).__name__,
        })
        logger.error("gateway.pipeline_failed", extra={
            "run_id": run_id, "error_type": type(exc).__name__,
        })


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health(request: Request) -> HealthResponse:
    with tracer.start_as_current_span("gateway.health"):
        return HealthResponse(timestamp=datetime.now(timezone.utc))


# ── Topics ────────────────────────────────────────────────────────────────────

@app.get("/api/topics", response_model=list[TopicResponse])
@limiter.limit("60/minute")
async def list_topics(request: Request) -> list[TopicResponse]:
    with tracer.start_as_current_span("gateway.topics.list") as span:
        span.set_attribute("user_id", USER_ID)
        repo: TopicRepository = request.app.state.topic_repo
        topics = repo.get_all_topics()
        user_topics = [t for t in topics if t.user_id == USER_ID]
        span.set_attribute("topic_count", len(user_topics))
        return [TopicResponse(**t.model_dump()) for t in user_topics]


@app.post("/api/topics", response_model=TopicResponse, status_code=201)
@limiter.limit("30/minute")
async def create_topic(request: Request, body: TopicCreate) -> TopicResponse:
    with tracer.start_as_current_span("gateway.topics.create") as span:
        span.set_attribute("user_id", USER_ID)
        repo: TopicRepository = request.app.state.topic_repo
        topic = TopicSubscription(
            topic_id=f"topic-{uuid.uuid4().hex[:12]}",
            user_id=USER_ID,
            topic_text=body.topic_text,
            constraints=body.constraints,
        )
        repo.save_topic(topic)
        logger.info("gateway.topic_created", extra={"topic_id": topic.topic_id})
        return TopicResponse(**topic.model_dump())


@app.put("/api/topics/{topic_id}", response_model=TopicResponse)
@limiter.limit("30/minute")
async def update_topic(request: Request, topic_id: str, body: TopicUpdate) -> TopicResponse:
    with tracer.start_as_current_span("gateway.topics.update") as span:
        span.set_attribute("topic_id", topic_id)
        repo: TopicRepository = request.app.state.topic_repo
        try:
            topic = repo.get_topic(topic_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Topic not found")

        if topic.user_id != USER_ID:
            raise HTTPException(status_code=403, detail="Forbidden")

        updates: dict[str, Any] = {}
        if body.topic_text is not None:
            updates["topic_text"] = body.topic_text
        if body.constraints is not None:
            updates["constraints"] = body.constraints
        if body.active is not None:
            updates["active"] = body.active

        if updates:
            repo.update_topic(topic_id, updates)

        updated = repo.get_topic(topic_id)
        return TopicResponse(**updated.model_dump())


@app.delete("/api/topics/{topic_id}", status_code=204)
@limiter.limit("30/minute")
async def delete_topic(request: Request, topic_id: str) -> None:
    with tracer.start_as_current_span("gateway.topics.delete") as span:
        span.set_attribute("topic_id", topic_id)
        repo: TopicRepository = request.app.state.topic_repo
        try:
            topic = repo.get_topic(topic_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Topic not found")

        if topic.user_id != USER_ID:
            raise HTTPException(status_code=403, detail="Forbidden")

        repo.delete_topic(topic_id)
        logger.info("gateway.topic_deleted", extra={"topic_id": topic_id})


# ── Digests ───────────────────────────────────────────────────────────────────

@app.get("/api/digests", response_model=list[DigestSummary])
@limiter.limit("30/minute")
async def list_digests(request: Request) -> list[DigestSummary]:
    with tracer.start_as_current_span("gateway.digests.list"):
        digest_repo: DigestRepository = request.app.state.digest_repo
        run_ids = digest_repo.list_digests()
        summaries: list[tuple[datetime, DigestSummary]] = []
        for run_id in run_ids:
            try:
                data = digest_repo.load(run_id)
                generated_at = _parse_dt(data.get("generated_at"))
                summaries.append((
                    generated_at or datetime.min.replace(tzinfo=timezone.utc),
                    DigestSummary(
                        run_id=run_id,
                        generated_at=generated_at,
                        passed_count=data.get("passed_count", 0),
                        tombstoned_count=data.get("tombstoned_count", 0),
                        overall_confidence=data.get("overall_confidence", "LOW"),
                        run_status=data.get("run_status", "unknown"),
                    ),
                ))
            except Exception:
                continue
        summaries.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in summaries]


@app.get("/api/digests/latest")
@limiter.limit("30/minute")
async def get_latest_digest(request: Request) -> dict:
    with tracer.start_as_current_span("gateway.digests.latest"):
        digest_repo: DigestRepository = request.app.state.digest_repo
        run_ids = digest_repo.list_digests()
        if not run_ids:
            raise HTTPException(status_code=404, detail="No digests found")

        # Load all to find the most recent by generated_at
        best: Optional[tuple[datetime, dict]] = None
        for run_id in run_ids:
            try:
                data = digest_repo.load(run_id)
                generated_at = _parse_dt(data.get("generated_at"))
                if generated_at is None:
                    continue
                if best is None or generated_at > best[0]:
                    best = (generated_at, data)
            except Exception:
                continue

        if best is None:
            raise HTTPException(status_code=404, detail="No digests found")
        return best[1]


@app.get("/api/digests/{run_id}")
@limiter.limit("30/minute")
async def get_digest(request: Request, run_id: str) -> dict:
    with tracer.start_as_current_span("gateway.digests.get") as span:
        span.set_attribute("run_id", run_id)
        try:
            return request.app.state.digest_repo.load(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Digest not found")


# ── Pipeline runs ─────────────────────────────────────────────────────────────

@app.post("/api/runs/trigger", response_model=RunTriggerResponse)
@limiter.limit("5/minute")
async def trigger_run(request: Request) -> RunTriggerResponse:
    with tracer.start_as_current_span("gateway.runs.trigger") as span:
        # Reject if a gateway-triggered run is already in flight
        active = next(
            (rid for rid, s in _run_status.items() if s["status"] in ("pending", "running")),
            None,
        )
        if active:
            raise HTTPException(
                status_code=409,
                detail={"message": "run_active", "active_run_id": active},
            )

        run_id = f"gw-{uuid.uuid4().hex[:12]}"
        span.set_attribute("run_id", run_id)

        _run_status[run_id] = {
            "run_id": run_id,
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "triggered_by": "api",
        }

        topic_repo: TopicRepository = request.app.state.topic_repo
        asyncio.create_task(_run_pipeline_task(run_id, topic_repo))

        logger.info("gateway.run_triggered", extra={"run_id": run_id})
        return RunTriggerResponse(run_id=run_id)


@app.get("/api/runs/{run_id}/status", response_model=RunStatusResponse)
@limiter.limit("60/minute")
async def get_run_status(request: Request, run_id: str) -> RunStatusResponse:
    with tracer.start_as_current_span("gateway.runs.status") as span:
        span.set_attribute("run_id", run_id)

        # Check in-memory store first (captures pending/running state)
        if run_id in _run_status:
            state = _run_status[run_id]
            return RunStatusResponse(
                run_id=run_id,
                status=state["status"],
                started_at=_parse_dt(state.get("started_at")),
                completed_at=_parse_dt(state.get("completed_at")),
                triggered_by=state.get("triggered_by", "api"),
                error=state.get("error"),
            )

        # Fall back to RunRepository (runs completed in a prior session)
        try:
            run_data = request.app.state.run_repo.load(run_id)
            return RunStatusResponse(
                run_id=run_id,
                status=run_data.get("status", "unknown"),
                completed_at=_parse_dt(run_data.get("completed_at")),
            )
        except KeyError:
            pass

        raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/runs/history", response_model=list[RunHistoryEntry])
@limiter.limit("30/minute")
async def get_run_history(request: Request, n: int = 20) -> list[RunHistoryEntry]:
    with tracer.start_as_current_span("gateway.runs.history") as span:
        n = min(max(n, 1), 100)
        span.set_attribute("limit", n)

        run_repo: RunRepository = request.app.state.run_repo
        digest_repo: DigestRepository = request.app.state.digest_repo

        # Collect entries: start with in-memory gateway runs
        entries: dict[str, RunHistoryEntry] = {}
        for run_id, state in _run_status.items():
            entries[run_id] = RunHistoryEntry(
                run_id=run_id,
                status=state["status"],
                started_at=_parse_dt(state.get("started_at")),
                completed_at=_parse_dt(state.get("completed_at")),
                passed_count=state.get("passed_count", 0),
                tombstoned_count=state.get("tombstoned_count", 0),
                triggered_by=state.get("triggered_by", "api"),
            )

        # Merge completed pipeline runs (from RunRepository — written by finalize_phase2)
        for run_id in run_repo.list_runs():
            if run_id in entries:
                continue
            try:
                run_data = run_repo.load(run_id)
                metrics = run_data.get("metrics", {})
                # Try to get pass/tombstone counts from digest if not in run metrics
                passed = metrics.get("passed_count", 0)
                tombstoned = metrics.get("tombstoned_count", 0)
                if passed == 0 and tombstoned == 0:
                    try:
                        digest_data = digest_repo.load(run_id)
                        passed = digest_data.get("passed_count", 0)
                        tombstoned = digest_data.get("tombstoned_count", 0)
                    except KeyError:
                        pass

                completed_at = _parse_dt(run_data.get("completed_at"))
                entries[run_id] = RunHistoryEntry(
                    run_id=run_id,
                    status=run_data.get("status", "unknown"),
                    completed_at=completed_at,
                    passed_count=passed,
                    tombstoned_count=tombstoned,
                    triggered_by="scheduler",
                )
            except Exception:
                continue

        # Sort by started_at desc (fall back to completed_at for runs without started_at)
        def sort_key(e: RunHistoryEntry) -> datetime:
            ts = e.started_at or e.completed_at
            return ts if ts else datetime.min.replace(tzinfo=timezone.utc)

        sorted_entries = sorted(entries.values(), key=sort_key, reverse=True)
        return sorted_entries[:n]


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post("/api/feedback/summary", response_model=FeedbackResponse, status_code=201)
@limiter.limit("30/minute")
async def feedback_summary(request: Request, body: FeedbackSummaryRequest) -> FeedbackResponse:
    with tracer.start_as_current_span("gateway.feedback.summary") as span:
        span.set_attribute("run_id", body.run_id)
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        request.app.state.feedback_repo.save(feedback_id, {
            "type": "summary_feedback",
            "feedback_id": feedback_id,
            "run_id": body.run_id,
            "article_url": body.article_url,
            "vote": body.vote,
            "note": body.note,
            "user_id": USER_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("gateway.feedback.summary_recorded", extra={
            "feedback_id": feedback_id, "run_id": body.run_id,
        })
        return FeedbackResponse(feedback_id=feedback_id)


@app.post("/api/feedback/topic", response_model=FeedbackResponse, status_code=201)
@limiter.limit("30/minute")
async def feedback_topic(request: Request, body: FeedbackTopicRequest) -> FeedbackResponse:
    with tracer.start_as_current_span("gateway.feedback.topic") as span:
        span.set_attribute("topic_id", body.topic_id)
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        request.app.state.feedback_repo.save(feedback_id, {
            "type": "topic_relevance",
            "feedback_id": feedback_id,
            "topic_id": body.topic_id,
            "relevance_score": body.relevance_score,
            "note": body.note,
            "user_id": USER_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return FeedbackResponse(feedback_id=feedback_id)


@app.post("/api/feedback/missed", response_model=FeedbackResponse, status_code=201)
@limiter.limit("30/minute")
async def feedback_missed(request: Request, body: FeedbackMissedRequest) -> FeedbackResponse:
    with tracer.start_as_current_span("gateway.feedback.missed"):
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        request.app.state.feedback_repo.save(feedback_id, {
            "type": "missed_topic",
            "feedback_id": feedback_id,
            "user_id": USER_ID,
            "note": body.note,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return FeedbackResponse(feedback_id=feedback_id)


@app.post("/api/feedback/override", response_model=FeedbackResponse, status_code=201)
@limiter.limit("30/minute")
async def feedback_override(request: Request, body: FeedbackOverrideRequest) -> FeedbackResponse:
    with tracer.start_as_current_span("gateway.feedback.override") as span:
        span.set_attribute("run_id", body.run_id)
        span.set_attribute("action", body.action)
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        request.app.state.feedback_repo.save(feedback_id, {
            "type": "candidate_override",
            "feedback_id": feedback_id,
            "run_id": body.run_id,
            "topic_id": body.topic_id,
            "action": body.action,
            "note": body.note,
            "user_id": USER_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("gateway.feedback.override_recorded", extra={
            "feedback_id": feedback_id,
            "run_id": body.run_id,
            "action": body.action,
        })
        return FeedbackResponse(feedback_id=feedback_id)
