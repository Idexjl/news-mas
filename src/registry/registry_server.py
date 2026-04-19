"""
Agent registry server.

Exposes AgentConfig CRUD over HTTP so agents can fetch their configuration at
startup (GET /agents/{id}/config) and so operators can update it at runtime
(PUT /agents/{id}/config — admin capability required).

[NETWORK-SEGMENTATION-TODO] In production, PUT /config endpoints must be
network-isolated — reachable only from the admin/orchestration subnet, NOT
from the compute subnet where worker agents run. Worker agents can GET their
own config but must never be able to reach PUT endpoints.
Azure equivalent: NSG inbound rule allowing PUT traffic only from the
orchestration subnet IP range; deny-all from compute subnet on PUT.
The application-layer capability check below is defence-in-depth only.
"""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.token_service import validate_token
from src.common.observability import get_logger, setup_telemetry
from src.registry import card_store
from src.registry.models import AgentConfig

setup_telemetry("registry")
logger = get_logger(__name__)

# Read once at startup; monkeypatch this in tests to control auth.
_SECRET_KEY = os.getenv("MAS_SECRET_KEY", "dev-secret")

limiter = Limiter(key_func=get_remote_address)
async def startup() -> None:
    """Seed default AgentConfig records. Called on startup and directly in tests."""
    existing = {c.agent_id for c in card_store.list_agent_configs()}
    seeded = 0
    for config in _DEFAULTS:
        if config.agent_id not in existing:
            card_store.save_agent_config(config)
            seeded += 1
    logger.info("registry_seeded", extra={"seeded": seeded, "total": len(_DEFAULTS)})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await startup()
    yield


app = FastAPI(title="Agent Registry", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Default configs seeded at startup ─────────────────────────────────────────

_DEFAULTS: list[AgentConfig] = [
    AgentConfig(
        agent_id="search-worker",
        model_provider="tavily",
        model_id="none",
        capabilities=["search.web"],
        token_budget=0,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=30,
    ),
    AgentConfig(
        agent_id="heat-scorer",
        model_provider="ollama",
        model_id="gemma4:e4b",
        capabilities=["score.heat"],
        token_budget=4096,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=60,
    ),
    AgentConfig(
        agent_id="filter-agent",
        model_provider="none",
        model_id="none",
        capabilities=["filter.articles"],
        token_budget=0,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=10,
    ),
    AgentConfig(
        agent_id="selector",
        model_provider="none",
        model_id="none",
        capabilities=["select.topk"],
        token_budget=0,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=10,
    ),
    AgentConfig(
        agent_id="phase1-judge",
        model_provider="ollama",
        model_id="gemma4:e4b",
        capabilities=["judge.phase1"],
        token_budget=8192,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=60,
    ),
    AgentConfig(
        agent_id="summarizer",
        model_provider="anthropic",
        model_id="claude-sonnet-4-6",
        capabilities=["summarize.article"],
        token_budget=16384,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase2"],
        network_tier="compute",
        timeout_seconds=120,
    ),
    AgentConfig(
        agent_id="reviewer",
        model_provider="anthropic",
        model_id="claude-sonnet-4-6",
        capabilities=["review.summary"],
        token_budget=8192,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase2"],
        network_tier="compute",
        timeout_seconds=120,
    ),
    AgentConfig(
        agent_id="relevance-gate",
        model_provider="ollama",
        model_id="gemma4:e4b",
        capabilities=["gate.relevance"],
        token_budget=4096,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase2"],
        network_tier="compute",
        timeout_seconds=60,
    ),
]


# ── Auth middleware ────────────────────────────────────────────────────────────

class SecurityHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public_paths = {"/health", "/config/defaults"}
        if request.url.path not in public_paths:
            token = request.headers.get("X-MAS-Secret", "")
            # Compare against the module-level _SECRET_KEY so tests can patch it.
            if not token or not hmac.compare_digest(
                token.encode(), _SECRET_KEY.encode()
            ):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid or missing X-MAS-Secret"}
                )
        return await call_next(request)


app.add_middleware(SecurityHeaderMiddleware)


def _require_config_write(authorization: str | None) -> None:
    """
    Verify the caller's AAP token includes registry.config.write.

    [NETWORK-SEGMENTATION-TODO] This capability check is defence-in-depth.
    Primary enforcement is network isolation: the compute subnet cannot reach
    PUT endpoints at the network level. This check adds an application-layer
    guard for defence in depth and audit purposes.

    Raises HTTPException(403) if the token is missing or lacks the capability.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=403,
            detail="Admin AAP token required in Authorization: Bearer <token>",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = validate_token(token, secret_key=_SECRET_KEY)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=f"Invalid AAP token: {exc}") from exc

    if "registry.config.write" not in claims.get("aap_capabilities", []):
        raise HTTPException(
            status_code=403,
            detail="Caller lacks registry.config.write capability. "
            "Worker agents cannot modify registry config.",
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "registry"}


@app.get("/config/defaults", response_model=list[AgentConfig])
async def get_defaults():
    """Return the built-in default configs for all agents (no auth required)."""
    return _DEFAULTS


@app.get("/agents/{agent_id}/config", response_model=AgentConfig)
@limiter.limit("60/minute")
async def get_agent_config(agent_id: str, request: Request):
    """
    Fetch an agent's config. Called by every agent at startup via bootstrap_agent().

    [SIEM-TODO] Each call to this endpoint is an agent "registration" event —
    an agent is announcing its presence and fetching its operating parameters.
    These events feed Sentinel UEBA to build a behavioral baseline per agent:
      - Expected call frequency: each agent calls once at startup. A second
        call from the same agent_id within the same deployment window is
        anomalous (possible restart loop or bootstrap failure storm).
      - Expected source subnet: the request source IP should be within the
        mas-compute or mas-orchestration subnet. A call from outside those
        ranges (especially from mas-identity or external) is a high-severity
        incident — the registry should not be reachable from outside the
        intended tiers (enforced by NSG; this log is defence-in-depth).
      - Expected agent_id set: Sentinel can alert on requests for agent IDs
        that do not exist in the seeded defaults — a probe for undocumented
        endpoints or a misconfigured agent.
      In production, emit a structured log event here:
        {"event": "agent_bootstrap", "agent_id": agent_id, "source_ip": ...}
      so Sentinel can join bootstrap events with the agent's first AAP token
      mint event (same agent_id, close timestamps) to confirm normal startup.
    """
    try:
        config = card_store.get_agent_config(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return config


@app.put("/agents/{agent_id}/config", response_model=AgentConfig)
@limiter.limit("30/minute")
async def put_agent_config(
    agent_id: str,
    request: Request,
    updates: dict,
    authorization: str | None = Header(default=None),
):
    """
    Update an agent's config. Admin capability required.

    [NETWORK-SEGMENTATION-TODO] In production this endpoint must only be
    reachable from the admin/orchestration subnet. NSG rules deny inbound
    traffic from the compute subnet on this path regardless of auth headers.
    """
    _require_config_write(authorization)

    updated_by = "api"
    try:
        auth_header = authorization or ""
        if auth_header.startswith("Bearer "):
            claims = validate_token(
                auth_header.removeprefix("Bearer ").strip(),
                secret_key=_SECRET_KEY,
            )
            updated_by = claims.get("sub", "api")
    except ValueError:
        pass

    try:
        config = card_store.update_agent_config(agent_id, updates, updated_by)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info(
        "config_updated",
        extra={"agent_id": agent_id, "updated_by": updated_by},
    )
    return config
