"""
Pipeline scheduler — APScheduler-based trigger for the news-mas pipeline.

Configuration (via .env):
  SCHEDULE_CRON             — standard cron expression (default: "0 8 * * 1")
  SCHEDULE_INTERVAL_HOURS   — interval in hours; when set, overrides SCHEDULE_CRON
  SCHEDULE_ENABLED          — "true" to activate scheduled runs (default: "true")
  SCHEDULE_TIMEZONE         — IANA timezone name (default: "America/Denver")
  SINCE_DATE_DEFAULT_DAYS   — article age cutoff in days (default: 7)
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel

from src.common.observability import get_logger, get_tracer, setup_metrics
from src.common.schemas import DigestOutput, TopicInput
from src.common.storage import DigestRepository, FernetStorage
from src.common.topic_repository import TopicRepository, TopicSubscription
from src.orchestrators.pipeline import run_full_pipeline

logger = get_logger(__name__)

# ── LangSmith parent run (optional) ──────────────────────────────────────────
try:
    from langsmith import traceable as _ls_traceable
except ImportError:
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

# ── APScheduler instance ──────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()

# ── Scheduler run state ───────────────────────────────────────────────────────
_SCHEDULER_RUNS = "scheduler_runs"
_STALE_RUN_HOURS = 2


class SchedulerRunState(BaseModel):
    run_id: str
    status: str  # running|complete|complete_no_topics|failed|interrupted|cancelled
    started_at: datetime
    triggered_by: str = "scheduler"
    completed_at: Optional[datetime] = None
    digest_id: Optional[str] = None
    error: Optional[str] = None


class SchedulerRunRepository:
    """Fernet-backed repository for scheduler run lifecycle records."""

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save_run(self, run: SchedulerRunState) -> None:
        self._store.save(_SCHEDULER_RUNS, run.run_id, run.model_dump(mode="json"))

    def get_run(self, run_id: str) -> SchedulerRunState:
        data = self._store.load(_SCHEDULER_RUNS, run_id)
        return SchedulerRunState(**data)

    def get_active_run(self) -> Optional[SchedulerRunState]:
        for rid in self._store.list(_SCHEDULER_RUNS):
            try:
                run = self.get_run(rid)
                if run.status == "running":
                    return run
            except Exception:
                pass
        return None

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        completed_at: Optional[datetime] = None,
        digest_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        run = self.get_run(run_id)
        run.status = status
        if completed_at is not None:
            run.completed_at = completed_at
        if digest_id is not None:
            run.digest_id = digest_id
        if error is not None:
            run.error = error
        self.save_run(run)

    def list_recent_runs(self, n: int = 10) -> list[SchedulerRunState]:
        """Return up to *n* runs sorted by started_at descending."""
        runs: list[SchedulerRunState] = []
        for rid in self._store.list(_SCHEDULER_RUNS):
            try:
                runs.append(self.get_run(rid))
            except Exception:
                pass
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:n]


# ── Module-level singletons (initialised by init_scheduler_dependencies) ──────
_run_repo: Optional[SchedulerRunRepository] = None
_topic_repo: Optional[TopicRepository] = None
_digest_store: Optional[DigestRepository] = None

# OTel metric instruments (initialised alongside dependencies)
_runs_counter = None
_duration_hist = None
_topics_counter = None
_tokens_hist = None
_pass_rate_hist = None


def init_scheduler_dependencies(storage: FernetStorage) -> None:
    """Initialise repositories and OTel metrics. Call once at service startup."""
    global _run_repo, _topic_repo, _digest_store
    _run_repo = SchedulerRunRepository(storage)
    _topic_repo = TopicRepository(storage)
    _digest_store = DigestRepository(storage)
    _init_metrics()


def _init_metrics() -> None:
    global _runs_counter, _duration_hist, _topics_counter, _tokens_hist, _pass_rate_hist
    meter = setup_metrics("news-mas-scheduler")
    _runs_counter = meter.create_counter(
        "mas.scheduler.runs_total",
        description="Total pipeline runs triggered by the scheduler",
    )
    _duration_hist = meter.create_histogram(
        "mas.scheduler.run_duration_ms",
        unit="ms",
        description="Pipeline run wall-clock duration",
    )
    _topics_counter = meter.create_counter(
        "mas.scheduler.topics_processed",
        description="Total topics processed across scheduler runs",
    )
    _tokens_hist = meter.create_histogram(
        "mas.scheduler.tokens_per_run",
        unit="tokens",
        description="Tokens consumed per pipeline run",
    )
    _pass_rate_hist = meter.create_histogram(
        "mas.scheduler.pass_rate",
        description="Fraction of topics that passed the relevance gate (0.0–1.0)",
    )


# ── Accessor helpers (allow api.py to read current singleton values) ───────────

def get_run_repo() -> Optional[SchedulerRunRepository]:
    return _run_repo


def get_topic_repo() -> Optional[TopicRepository]:
    return _topic_repo


def get_digest_store() -> Optional[DigestRepository]:
    return _digest_store


# ── Internal helpers ──────────────────────────────────────────────────────────

def _generate_run_id() -> str:
    return f"sched-{uuid.uuid4().hex[:12]}"


def _topics_to_inputs(topics: list[TopicSubscription]) -> list[TopicInput]:
    return [
        TopicInput(
            topic_id=t.topic_id,
            topic_text=t.topic_text,
            constraints=t.constraints,
        )
        for t in topics
    ]


def emit_run_complete_metrics(digest: DigestOutput) -> None:
    """Record OTel metrics after a successful pipeline run."""
    if _runs_counter:
        _runs_counter.add(1, {"status": "complete"})
    if _duration_hist and digest.run_duration_ms:
        _duration_hist.record(digest.run_duration_ms)
    if _topics_counter and digest.total_topics:
        _topics_counter.add(digest.total_topics)
    if _tokens_hist and digest.total_tokens_used:
        _tokens_hist.record(digest.total_tokens_used)
    if _pass_rate_hist and digest.total_topics:
        _pass_rate_hist.record(digest.passed_count / digest.total_topics)


# ── LangSmith-traced pipeline wrapper ────────────────────────────────────────

@_ls_traceable(name="scheduled-pipeline-run", run_type="chain")
async def _run_pipeline_traced(
    topics: list[TopicInput],
    since_date: str,
    user_id: str,
    run_id: str,
    triggered_by: str,
    topic_count: int,
) -> DigestOutput:
    """
    Thin wrapper so every pipeline invocation appears as a top-level
    LangSmith run named "scheduled-pipeline-run". Phase 1 and Phase 2
    child traces nest under it automatically via langsmith context propagation.
    """
    return await run_full_pipeline(
        topics=topics,
        since_date=since_date,
        user_id=user_id,
        run_id=run_id,
    )


# ── Core execution ────────────────────────────────────────────────────────────

async def execute_run(run_id: str, triggered_by: str) -> None:
    """
    Core execution path shared by the scheduled trigger and the manual API.

    Saves the run record, executes the full pipeline, persists the digest,
    and updates the final run status. Does NOT perform a circuit-breaker
    check — callers are responsible for that guard.
    """
    tracer = get_tracer("news-mas-scheduler")

    with tracer.start_as_current_span("scheduler.trigger") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("triggered_by", triggered_by)

        try:
            _run_repo.save_run(SchedulerRunState(
                run_id=run_id,
                status="running",
                started_at=datetime.now(UTC),
                triggered_by=triggered_by,
            ))

            topics = _topic_repo.get_all_topics()
            span.set_attribute("topic_count", len(topics))

            if not topics:
                logger.info("scheduler.no_topics", extra={"run_id": run_id})
                _run_repo.update_status(run_id, "complete_no_topics")
                if _runs_counter:
                    _runs_counter.add(1, {"status": "complete_no_topics"})
                return

            since_days = int(os.getenv("SINCE_DATE_DEFAULT_DAYS", "7"))
            since_date = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
            span.set_attribute("since_date", since_date)

            topic_inputs = _topics_to_inputs(topics)

            with tracer.start_as_current_span("scheduler.pipeline") as pipeline_span:
                pipeline_span.set_attribute("run_id", run_id)
                digest = await _run_pipeline_traced(
                    topics=topic_inputs,
                    since_date=since_date,
                    user_id="system",
                    run_id=run_id,
                    triggered_by=triggered_by,
                    topic_count=len(topics),
                )
                pipeline_span.set_attribute("duration_ms", digest.run_duration_ms)

            _digest_store.save(digest.run_id, digest.model_dump(mode="json"))

            _run_repo.update_status(
                run_id,
                "complete",
                completed_at=datetime.now(UTC),
                digest_id=digest.run_id,
            )

            emit_run_complete_metrics(digest)

            span.add_event("scheduler.run_complete", {
                "run_id": run_id,
                "passed": digest.passed_count,
                "tombstoned": digest.tombstoned_count,
                "tokens": digest.total_tokens_used,
                "duration_ms": digest.run_duration_ms,
            })
            logger.info("scheduler.run_complete", extra={
                "run_id": run_id,
                "passed": digest.passed_count,
                "tombstoned": digest.tombstoned_count,
                "tokens": digest.total_tokens_used,
                "duration_ms": digest.run_duration_ms,
            })

        except Exception as e:
            _run_repo.update_status(run_id, "failed", error=type(e).__name__)
            span.add_event("scheduler.run_failed", {
                "run_id": run_id,
                "error_type": type(e).__name__,
            })
            if _runs_counter:
                _runs_counter.add(1, {"status": "failed"})
            logger.error("scheduler.run_failed", extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
            })
            raise


async def trigger_pipeline_run() -> None:
    """
    APScheduler job — called on the configured schedule.

    Enforces the circuit breaker (max_instances=1 in APScheduler plus an
    explicit active-run check) before delegating to execute_run.
    """
    if _run_repo is None:
        logger.warning("scheduler.not_initialized")
        return

    tracer = get_tracer("news-mas-scheduler")

    active_run = _run_repo.get_active_run()
    if active_run:
        with tracer.start_as_current_span("scheduler.trigger") as span:
            span.add_event("scheduler.skipped", {
                "reason": "run_active",
                "active_run_id": active_run.run_id,
            })
        logger.warning(
            "scheduler.skipped.run_active",
            extra={"active_run_id": active_run.run_id},
        )
        return

    run_id = _generate_run_id()
    await execute_run(run_id, "scheduler")


# ── Startup cleanup ───────────────────────────────────────────────────────────

async def cleanup_interrupted_runs() -> None:
    """
    Called at startup. Runs with status='running' that started more than
    _STALE_RUN_HOURS ago are marked 'interrupted' (presumed dead from a
    previous crash). Recent running runs are logged as a warning but left
    untouched — they may belong to a legitimately running pipeline.
    """
    if _run_repo is None:
        return

    tracer = get_tracer("news-mas-scheduler")
    with tracer.start_as_current_span("scheduler.cleanup"):
        stale_before = datetime.now(UTC) - timedelta(hours=_STALE_RUN_HOURS)
        for run in _run_repo.list_recent_runs(n=50):
            if run.status != "running":
                continue
            started = run.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            if started < stale_before:
                _run_repo.update_status(run.run_id, "interrupted")
                logger.warning("scheduler.run_interrupted", extra={
                    "run_id": run.run_id,
                    "started_at": run.started_at.isoformat(),
                })
            else:
                logger.warning("scheduler.run_in_progress_on_startup", extra={
                    "run_id": run.run_id,
                })


# ── Trigger builder ───────────────────────────────────────────────────────────

def build_trigger():
    """Build the APScheduler trigger from environment configuration."""
    tz = os.getenv("SCHEDULE_TIMEZONE", "America/Denver")
    interval_hours = os.getenv("SCHEDULE_INTERVAL_HOURS", "").strip()
    if interval_hours:
        return IntervalTrigger(hours=int(interval_hours))
    cron = os.getenv("SCHEDULE_CRON", "0 8 * * 1")
    return CronTrigger.from_crontab(cron, timezone=tz)
