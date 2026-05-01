"""
Scheduler management API — FastAPI app for manual pipeline control.

Endpoints
  POST   /runs/trigger           — manual run trigger (requires schedule.trigger)
  GET    /runs/status/{run_id}   — run state lookup
  GET    /runs/latest            — most recent run summary with digest stats
  GET    /runs/history           — last N runs (default 10, max 100)
  GET    /health                 — scheduler status and next scheduled run
  POST   /runs/{run_id}/cancel   — cancel a running pipeline (requires schedule.trigger)

All endpoints are rate-limited. Auth is checked on trigger and cancel only.
"""
from __future__ import annotations

import os

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.token_service import validate_token
from src.common.observability import configure_langsmith, get_logger, setup_telemetry
from src.common.storage import FernetStorage

from .scheduler import (
    SchedulerRunState,
    _generate_run_id,
    build_trigger,
    cleanup_interrupted_runs,
    execute_run,
    get_digest_store,
    get_run_repo,
    init_scheduler_dependencies,
    scheduler,
    trigger_pipeline_run,
)
from datetime import UTC, datetime

setup_telemetry("news-mas-scheduler")
configure_langsmith()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Scheduler API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Security headers middleware ────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ── AAP token auth dependency ─────────────────────────────────────────────────

def _require_schedule_trigger(request: Request) -> dict:
    """FastAPI dependency: validate AAP token with schedule.trigger capability."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth[7:]
    secret = os.getenv("MAS_SECRET_KEY", "")
    try:
        payload = validate_token(token, secret_key=secret)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if "schedule.trigger" not in payload.get("aap_capabilities", []):
        raise HTTPException(status_code=403, detail="Missing schedule.trigger capability")
    return payload


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    storage = FernetStorage()
    init_scheduler_dependencies(storage)
    await cleanup_interrupted_runs()

    if os.getenv("SCHEDULE_ENABLED", "true").lower() == "true":
        trigger = build_trigger()
        scheduler.add_job(
            trigger_pipeline_run,
            trigger=trigger,
            id="pipeline_trigger",
            max_instances=1,
            replace_existing=True,
        )
        scheduler.start()
        logger.info("scheduler.started", extra={
            "cron": os.getenv("SCHEDULE_CRON", "0 8 * * 1"),
            "timezone": os.getenv("SCHEDULE_TIMEZONE", "America/Denver"),
            "interval_hours": os.getenv("SCHEDULE_INTERVAL_HOURS", ""),
        })
    else:
        logger.info("scheduler.disabled")


@app.on_event("shutdown")
async def shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler.stopped")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    next_run_at = None
    jobs = scheduler.get_jobs() if scheduler.running else []
    if jobs:
        nrt = jobs[0].next_run_time
        if nrt:
            next_run_at = nrt.isoformat()

    last_run_id = None
    rr = get_run_repo()
    if rr:
        recent = rr.list_recent_runs(n=1)
        if recent:
            last_run_id = recent[0].run_id

    return {
        "status": "ok",
        "next_run_at": next_run_at,
        "schedule_config": {
            "cron": os.getenv("SCHEDULE_CRON", "0 8 * * 1"),
            "interval_hours": os.getenv("SCHEDULE_INTERVAL_HOURS", ""),
            "timezone": os.getenv("SCHEDULE_TIMEZONE", "America/Denver"),
            "enabled": os.getenv("SCHEDULE_ENABLED", "true"),
        },
        "last_run_id": last_run_id,
    }


@app.post("/runs/trigger")
@limiter.limit("10/minute")
async def trigger_run(
    request: Request,
    background_tasks: BackgroundTasks,
    _auth: dict = Depends(_require_schedule_trigger),
):
    rr = get_run_repo()
    if rr is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")

    active = rr.get_active_run()
    if active:
        return JSONResponse(
            status_code=409,
            content={"detail": "run_active", "active_run_id": active.run_id},
        )

    run_id = _generate_run_id()
    background_tasks.add_task(execute_run, run_id, "api")
    logger.info("scheduler.manual_trigger", extra={"run_id": run_id})
    return {"run_id": run_id, "status": "triggered"}


@app.get("/runs/status/{run_id}")
@limiter.limit("60/minute")
async def get_run_status(request: Request, run_id: str):
    rr = get_run_repo()
    if rr is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")
    try:
        run = rr.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.model_dump(mode="json")


@app.get("/runs/latest")
@limiter.limit("60/minute")
async def get_latest_run(request: Request):
    rr = get_run_repo()
    if rr is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")

    recent = rr.list_recent_runs(n=1)
    if not recent:
        return {"detail": "no_runs"}

    run = recent[0]
    result: dict = {
        "run_id": run.run_id,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "triggered_by": run.triggered_by,
    }

    ds = get_digest_store()
    if run.digest_id and ds:
        try:
            digest_data = ds.load(run.digest_id)
            result.update({
                "passed_count": digest_data.get("passed_count", 0),
                "tombstoned_count": digest_data.get("tombstoned_count", 0),
                "generated_at": digest_data.get("generated_at"),
            })
        except KeyError:
            pass

    return result


@app.get("/runs/history")
@limiter.limit("30/minute")
async def get_run_history(request: Request, n: int = 10):
    rr = get_run_repo()
    if rr is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")
    runs = rr.list_recent_runs(n=min(n, 100))
    return [r.model_dump(mode="json") for r in runs]


@app.post("/runs/{run_id}/cancel")
@limiter.limit("10/minute")
async def cancel_run(
    request: Request,
    run_id: str,
    _auth: dict = Depends(_require_schedule_trigger),
):
    rr = get_run_repo()
    if rr is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")

    try:
        run = rr.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Run is not active (status={run.status})",
        )

    rr.update_status(run_id, "cancelled")
    logger.info("scheduler.run_cancelled", extra={"run_id": run_id})
    return {"run_id": run_id, "status": "cancelled"}
