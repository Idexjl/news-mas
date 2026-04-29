"""
Session-scoped fixtures for integration tests.

Cleans up runs stuck in "running" state from previous test sessions (e.g. from a
crash where finalize_phase1 never saved the completed status).  Without this,
the circuit breaker would block every new run on the next test session.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

_STALE_THRESHOLD_MINUTES = 5


@pytest.fixture(scope="session", autouse=True)
def cleanup_stale_runs():
    """Mark runs older than 2 hours that are still 'running' as 'aborted'."""
    fernet_key = os.getenv("FERNET_KEY")
    if not fernet_key:
        return

    try:
        from src.common.storage import FernetStorage, RunRepository

        repo = RunRepository(FernetStorage())
        now = datetime.now(timezone.utc)

        for run_id in repo.list_runs():
            try:
                run = repo.load(run_id)
                if run.get("status") != "running":
                    continue
                started_at = run.get("started_at")
                if not started_at:
                    continue
                started = datetime.fromisoformat(started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age_minutes = (now - started).total_seconds() / 60
                if age_minutes > _STALE_THRESHOLD_MINUTES:
                    repo.save(run_id, {**run, "status": "aborted"})
            except Exception:
                continue
    except Exception:
        pass
