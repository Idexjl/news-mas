"""
Unit tests for the scheduler subsystem.

Tests are isolated: all external dependencies (pipeline, storage, OTel metrics)
are mocked. No real pipeline runs are triggered.

Run with:
    pytest tests/test_scheduler.py -v
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src.auth.token_service import mint_token
from src.common.storage import FernetStorage
from src.common.topic_repository import TopicRepository, TopicSubscription
from src.scheduler.scheduler import (
    SchedulerRunRepository,
    SchedulerRunState,
    _generate_run_id,
    build_trigger,
    cleanup_interrupted_runs,
    execute_run,
    init_scheduler_dependencies,
    trigger_pipeline_run,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def fernet_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture()
def storage(tmp_path: Path, fernet_key: str) -> FernetStorage:
    return FernetStorage(data_dir=tmp_path, fernet_key=fernet_key)


@pytest.fixture()
def run_repo(storage: FernetStorage) -> SchedulerRunRepository:
    return SchedulerRunRepository(storage)


@pytest.fixture()
def topic_repo(storage: FernetStorage) -> TopicRepository:
    return TopicRepository(storage)


@pytest.fixture()
def _init_deps(storage: FernetStorage):
    """Initialise scheduler module globals with test storage."""
    with patch("src.scheduler.scheduler._runs_counter", None), \
         patch("src.scheduler.scheduler._duration_hist", None), \
         patch("src.scheduler.scheduler._topics_counter", None), \
         patch("src.scheduler.scheduler._tokens_hist", None), \
         patch("src.scheduler.scheduler._pass_rate_hist", None):
        init_scheduler_dependencies(storage)
        yield


@pytest.fixture()
def aap_token() -> str:
    secret = os.getenv("MAS_SECRET_KEY", "local-dev-only")
    return mint_token(
        subject="test-client",
        agent_name="test-scheduler",
        model_provider="none",
        model_id="none",
        capabilities=["schedule.trigger"],
        task_id="test-task",
        task_description="test",
        data_sensitivity="low",
        oversight_required=False,
        oversight_level="none",
        context={},
        secret_key=secret,
    )


@pytest.fixture()
def aap_token_no_caps() -> str:
    secret = os.getenv("MAS_SECRET_KEY", "local-dev-only")
    return mint_token(
        subject="test-client",
        agent_name="test-scheduler",
        model_provider="none",
        model_id="none",
        capabilities=["read.only"],
        task_id="test-task",
        task_description="test",
        data_sensitivity="low",
        oversight_required=False,
        oversight_level="none",
        context={},
        secret_key=secret,
    )


# ── SchedulerRunRepository tests ──────────────────────────────────────────────

def test_run_repo_save_and_get(run_repo):
    run = SchedulerRunState(
        run_id="r-001",
        status="running",
        started_at=datetime.now(UTC),
        triggered_by="scheduler",
    )
    run_repo.save_run(run)
    loaded = run_repo.get_run("r-001")
    assert loaded.run_id == "r-001"
    assert loaded.status == "running"
    assert loaded.triggered_by == "scheduler"


def test_run_repo_get_active_run_returns_running(run_repo):
    run = SchedulerRunState(run_id="r-active", status="running", started_at=datetime.now(UTC))
    run_repo.save_run(run)
    run_repo.save_run(SchedulerRunState(run_id="r-done", status="complete", started_at=datetime.now(UTC)))
    active = run_repo.get_active_run()
    assert active is not None
    assert active.run_id == "r-active"


def test_run_repo_get_active_run_none_when_all_complete(run_repo):
    run_repo.save_run(SchedulerRunState(run_id="r-1", status="complete", started_at=datetime.now(UTC)))
    run_repo.save_run(SchedulerRunState(run_id="r-2", status="failed", started_at=datetime.now(UTC)))
    assert run_repo.get_active_run() is None


def test_run_repo_update_status(run_repo):
    run = SchedulerRunState(run_id="r-upd", status="running", started_at=datetime.now(UTC))
    run_repo.save_run(run)
    run_repo.update_status("r-upd", "complete", completed_at=datetime.now(UTC), digest_id="d-123")
    loaded = run_repo.get_run("r-upd")
    assert loaded.status == "complete"
    assert loaded.digest_id == "d-123"
    assert loaded.completed_at is not None


def test_run_repo_list_recent_runs_sorted_desc(run_repo):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        run_repo.save_run(SchedulerRunState(
            run_id=f"r-{i:03d}",
            status="complete",
            started_at=t0 + timedelta(hours=i),
        ))
    recent = run_repo.list_recent_runs(n=3)
    assert len(recent) == 3
    assert recent[0].run_id == "r-004"
    assert recent[1].run_id == "r-003"
    assert recent[2].run_id == "r-002"


# ── trigger_pipeline_run — circuit breaker ────────────────────────────────────

async def test_trigger_skips_if_run_active(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod
    active = SchedulerRunState(run_id="r-active", status="running", started_at=datetime.now(UTC))
    sched_mod._run_repo.save_run(active)

    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock) as mock_pipeline:
        await trigger_pipeline_run()
        mock_pipeline.assert_not_called()


async def test_trigger_proceeds_when_no_active_run(storage, tmp_path):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    fake_digest = MagicMock()
    fake_digest.run_id = "sched-fake"
    fake_digest.passed_count = 2
    fake_digest.tombstoned_count = 0
    fake_digest.total_tokens_used = 100
    fake_digest.run_duration_ms = 500
    fake_digest.total_topics = 2
    fake_digest.model_dump.return_value = {}

    sched_mod._topic_repo.save_topic(TopicSubscription(
        topic_id="t-1", user_id="u-1", topic_text="test topic",
    ))

    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock, return_value=fake_digest):
        await trigger_pipeline_run()

    active = sched_mod._run_repo.get_active_run()
    assert active is None
    latest = sched_mod._run_repo.list_recent_runs(n=1)
    assert latest[0].status == "complete"


# ── No topics → complete_no_topics ───────────────────────────────────────────

async def test_no_topics_sets_complete_no_topics(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock) as mock_pipeline:
        await trigger_pipeline_run()
        mock_pipeline.assert_not_called()

    runs = sched_mod._run_repo.list_recent_runs(n=1)
    assert runs[0].status == "complete_no_topics"


# ── Error captured as error_type, not message ─────────────────────────────────

async def test_error_captured_as_type_not_message(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    sched_mod._topic_repo.save_topic(TopicSubscription(
        topic_id="t-err", user_id="u-1", topic_text="test",
    ))

    sensitive_msg = "PHI: patient John Doe SSN 123-45-6789"
    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock,
               side_effect=ValueError(sensitive_msg)):
        with pytest.raises(ValueError):
            await trigger_pipeline_run()

    runs = sched_mod._run_repo.list_recent_runs(n=1)
    assert runs[0].status == "failed"
    assert runs[0].error == "ValueError"
    # Confirm the sensitive message is NOT stored
    assert sensitive_msg not in (runs[0].error or "")


# ── Run status transitions ────────────────────────────────────────────────────

async def test_status_transitions_running_to_complete(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    sched_mod._topic_repo.save_topic(TopicSubscription(
        topic_id="t-tr", user_id="u-1", topic_text="test",
    ))

    fake_digest = MagicMock()
    fake_digest.run_id = "sched-tr"
    fake_digest.passed_count = 1
    fake_digest.tombstoned_count = 0
    fake_digest.total_tokens_used = 50
    fake_digest.run_duration_ms = 200
    fake_digest.total_topics = 1
    fake_digest.model_dump.return_value = {}

    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock, return_value=fake_digest):
        await trigger_pipeline_run()

    runs = sched_mod._run_repo.list_recent_runs(n=1)
    run = runs[0]
    assert run.status == "complete"
    assert run.completed_at is not None
    assert run.digest_id == "sched-tr"


# ── Interrupted run cleanup ───────────────────────────────────────────────────

async def test_cleanup_marks_stale_running_as_interrupted(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    stale_start = datetime.now(UTC) - timedelta(hours=3)
    stale_run = SchedulerRunState(
        run_id="r-stale",
        status="running",
        started_at=stale_start,
    )
    sched_mod._run_repo.save_run(stale_run)

    await cleanup_interrupted_runs()

    updated = sched_mod._run_repo.get_run("r-stale")
    assert updated.status == "interrupted"


async def test_cleanup_leaves_recent_running_untouched(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    recent_start = datetime.now(UTC) - timedelta(minutes=30)
    recent_run = SchedulerRunState(
        run_id="r-recent",
        status="running",
        started_at=recent_start,
    )
    sched_mod._run_repo.save_run(recent_run)

    await cleanup_interrupted_runs()

    unchanged = sched_mod._run_repo.get_run("r-recent")
    assert unchanged.status == "running"


# ── Schedule config from env ──────────────────────────────────────────────────

def test_build_trigger_uses_cron_by_default(monkeypatch):
    monkeypatch.delenv("SCHEDULE_INTERVAL_HOURS", raising=False)
    monkeypatch.setenv("SCHEDULE_CRON", "0 9 * * 2")
    monkeypatch.setenv("SCHEDULE_TIMEZONE", "UTC")
    from apscheduler.triggers.cron import CronTrigger
    trigger = build_trigger()
    assert isinstance(trigger, CronTrigger)


def test_build_trigger_uses_interval_when_set(monkeypatch):
    monkeypatch.setenv("SCHEDULE_INTERVAL_HOURS", "4")
    from apscheduler.triggers.interval import IntervalTrigger
    trigger = build_trigger()
    assert isinstance(trigger, IntervalTrigger)


def test_generate_run_id_has_sched_prefix():
    rid = _generate_run_id()
    assert rid.startswith("sched-")
    assert len(rid) > 10


# ── Metrics emitted on completion ─────────────────────────────────────────────

def test_emit_metrics_called_on_complete(storage):
    init_scheduler_dependencies(storage)
    import src.scheduler.scheduler as sched_mod

    fake_counter = MagicMock()
    fake_hist = MagicMock()

    sched_mod._runs_counter = fake_counter
    sched_mod._duration_hist = fake_hist
    sched_mod._topics_counter = MagicMock()
    sched_mod._tokens_hist = MagicMock()
    sched_mod._pass_rate_hist = MagicMock()

    digest = MagicMock()
    digest.run_id = "d-1"
    digest.passed_count = 3
    digest.tombstoned_count = 1
    digest.total_tokens_used = 200
    digest.run_duration_ms = 1000
    digest.total_topics = 4

    from src.scheduler.scheduler import emit_run_complete_metrics
    emit_run_complete_metrics(digest)

    fake_counter.add.assert_called_once_with(1, {"status": "complete"})
    fake_hist.record.assert_called_once_with(1000)


# ── API auth tests ────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client(storage, fernet_key, monkeypatch):
    monkeypatch.setenv("FERNET_KEY", fernet_key)
    monkeypatch.setenv("MAS_SECRET_KEY", "local-dev-only")
    monkeypatch.setenv("SCHEDULE_ENABLED", "false")  # don't start the real scheduler
    init_scheduler_dependencies(storage)
    from src.scheduler.api import app
    return TestClient(app, raise_server_exceptions=False)


def test_trigger_requires_bearer_token(api_client):
    resp = api_client.post("/runs/trigger")
    assert resp.status_code == 401


def test_trigger_rejects_invalid_token(api_client):
    resp = api_client.post(
        "/runs/trigger",
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert resp.status_code == 401


def test_trigger_rejects_token_without_capability(api_client, aap_token_no_caps):
    resp = api_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {aap_token_no_caps}"},
    )
    assert resp.status_code == 403


def test_trigger_returns_409_when_run_active(api_client, aap_token, storage):
    import src.scheduler.scheduler as sched_mod
    sched_mod._run_repo.save_run(SchedulerRunState(
        run_id="r-busy", status="running", started_at=datetime.now(UTC),
    ))
    resp = api_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {aap_token}"},
    )
    assert resp.status_code == 409
    assert "active_run_id" in resp.json()


def test_trigger_accepted_with_valid_token_and_no_active_run(api_client, aap_token):
    with patch("src.scheduler.scheduler._run_pipeline_traced", new_callable=AsyncMock):
        resp = api_client.post(
            "/runs/trigger",
            headers={"Authorization": f"Bearer {aap_token}"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "triggered"


def test_health_returns_expected_shape(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "schedule_config" in body
    assert "last_run_id" in body


def test_get_run_status_404_for_unknown(api_client):
    resp = api_client.get("/runs/status/nonexistent-run")
    assert resp.status_code == 404


def test_get_run_history_empty(api_client):
    resp = api_client.get("/runs/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_cancel_requires_auth(api_client):
    resp = api_client.post("/runs/r-001/cancel")
    assert resp.status_code == 401


def test_cancel_404_for_unknown_run(api_client, aap_token):
    resp = api_client.post(
        "/runs/nonexistent/cancel",
        headers={"Authorization": f"Bearer {aap_token}"},
    )
    assert resp.status_code == 404


def test_cancel_sets_status_cancelled(api_client, aap_token, storage):
    import src.scheduler.scheduler as sched_mod
    sched_mod._run_repo.save_run(SchedulerRunState(
        run_id="r-to-cancel", status="running", started_at=datetime.now(UTC),
    ))
    resp = api_client.post(
        "/runs/r-to-cancel/cancel",
        headers={"Authorization": f"Bearer {aap_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    updated = sched_mod._run_repo.get_run("r-to-cancel")
    assert updated.status == "cancelled"
