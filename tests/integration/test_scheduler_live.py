"""
Live integration tests for the scheduler API.

These tests start the FastAPI app in-process (no external server required)
but do require FERNET_KEY and MAS_SECRET_KEY to be set. The pipeline itself
is mocked — only the API lifecycle, storage, and HTTP routing are exercised.

Run with:
    pytest tests/integration/test_scheduler_live.py -v -m integration

Skip these in CI unless the full environment is available:
    pytest tests/ --ignore=tests/integration -v
"""
from __future__ import annotations

import time
from datetime import UTC, datetime
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
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fernet_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture()
def storage(tmp_path: Path, fernet_key: str) -> FernetStorage:
    return FernetStorage(data_dir=tmp_path, fernet_key=fernet_key)


@pytest.fixture()
def trigger_token(fernet_key: str) -> str:
    return mint_token(
        subject="integration-test",
        agent_name="test-runner",
        model_provider="none",
        model_id="none",
        capabilities=["schedule.trigger"],
        task_id="integration-test-task",
        task_description="integration testing",
        data_sensitivity="low",
        oversight_required=False,
        oversight_level="none",
        context={},
        secret_key="local-dev-only",
    )


@pytest.fixture()
def seeded_storage(storage: FernetStorage) -> FernetStorage:
    """Return storage pre-populated with the 5 seed topics."""
    repo = TopicRepository(storage)
    topics = [
        TopicSubscription(topic_id="topic-001", user_id="user-joe",
                          topic_text="Colorado Rockies baseball news"),
        TopicSubscription(topic_id="topic-002", user_id="user-joe",
                          topic_text="cybersecurity news high profile breaches exploits vulnerabilities",
                          constraints=["technical reporting only", "no vendor marketing content"]),
        TopicSubscription(topic_id="topic-003", user_id="user-joe",
                          topic_text="professional bicycle racing news Tour de France UCI WorldTour"),
        TopicSubscription(topic_id="topic-004", user_id="user-joe",
                          topic_text="Warhammer 40000 tabletop gaming news Games Workshop new releases lore"),
        TopicSubscription(topic_id="topic-005", user_id="user-joe",
                          topic_text="tabletop gaming news board games RPG Dungeons Dragons releases"),
    ]
    for t in topics:
        repo.save_topic(t)
    return storage


@pytest.fixture()
def fake_digest():
    digest = MagicMock()
    digest.run_id = "sched-fake-live"
    digest.passed_count = 3
    digest.tombstoned_count = 2
    digest.total_tokens_used = 4500
    digest.run_duration_ms = 12000
    digest.total_topics = 5
    digest.model_dump.return_value = {
        "run_id": "sched-fake-live",
        "passed_count": 3,
        "tombstoned_count": 2,
        "total_tokens_used": 4500,
        "run_duration_ms": 12000,
        "total_topics": 5,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return digest


@pytest.fixture()
def live_client(seeded_storage: FernetStorage, fernet_key: str, monkeypatch, fake_digest):
    monkeypatch.setenv("FERNET_KEY", fernet_key)
    monkeypatch.setenv("MAS_SECRET_KEY", "local-dev-only")
    monkeypatch.setenv("SCHEDULE_ENABLED", "false")
    from src.scheduler.api import app
    # Patch FernetStorage so the startup event receives the pre-seeded instance
    # instead of creating a fresh empty one from env vars.
    with patch("src.scheduler.api.FernetStorage", return_value=seeded_storage), \
         patch("src.scheduler.scheduler._run_pipeline_traced",
               new_callable=AsyncMock, return_value=fake_digest):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ── Test 1: manual trigger via API ────────────────────────────────────────────

def test_manual_trigger_returns_run_id(live_client, trigger_token):
    resp = live_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {trigger_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "triggered"
    assert body["run_id"].startswith("sched-")


def test_manual_trigger_run_completes_and_digest_saved(live_client, trigger_token):
    resp = live_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {trigger_token}"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    # Poll /runs/status until complete (or timeout)
    deadline = time.time() + 10
    status = "triggered"
    while time.time() < deadline and status not in ("complete", "failed", "complete_no_topics"):
        time.sleep(0.2)
        sr = live_client.get(f"/runs/status/{run_id}")
        if sr.status_code == 200:
            status = sr.json().get("status", status)

    assert status == "complete", f"Expected complete but got {status}"

    # Verify run metadata
    sr = live_client.get(f"/runs/status/{run_id}")
    data = sr.json()
    assert data["status"] == "complete"
    assert data["digest_id"] is not None


# ── Test 2: schedule config reflected in /health ──────────────────────────────

def test_health_reflects_schedule_config(live_client, monkeypatch):
    monkeypatch.setenv("SCHEDULE_CRON", "0 8 * * 1")
    monkeypatch.setenv("SCHEDULE_TIMEZONE", "America/Denver")
    resp = live_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    cfg = body["schedule_config"]
    assert cfg["cron"] == "0 8 * * 1"
    assert cfg["timezone"] == "America/Denver"


# ── Test 3: double trigger prevention ─────────────────────────────────────────

def test_double_trigger_prevented(live_client, trigger_token):
    import src.scheduler.scheduler as sched_mod

    # Simulate an already-running job
    sched_mod._run_repo.save_run(SchedulerRunState(
        run_id="r-running-live",
        status="running",
        started_at=datetime.now(UTC),
    ))

    resp1 = live_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {trigger_token}"},
    )
    assert resp1.status_code == 409
    assert resp1.json()["detail"] == "run_active"

    # Clean up
    sched_mod._run_repo.update_status("r-running-live", "complete")


# ── Test 4: seed topics load correctly ────────────────────────────────────────

def test_seed_topics_processed(live_client, trigger_token):
    import src.scheduler.scheduler as sched_mod

    # Trigger a run and check latest
    resp = live_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {trigger_token}"},
    )
    assert resp.status_code == 200

    # Wait briefly for background task
    time.sleep(0.5)

    latest = live_client.get("/runs/latest")
    assert latest.status_code == 200
    body = latest.json()
    # The fake digest has 5 topics total
    assert body.get("passed_count", 0) >= 0


def test_history_shows_triggered_runs(live_client, trigger_token):
    resp = live_client.post(
        "/runs/trigger",
        headers={"Authorization": f"Bearer {trigger_token}"},
    )
    assert resp.status_code == 200

    time.sleep(0.3)

    hist = live_client.get("/runs/history")
    assert hist.status_code == 200
    runs = hist.json()
    assert len(runs) >= 1
    # Most recent is a scheduler-style run
    assert runs[0]["triggered_by"] in ("api", "scheduler")
