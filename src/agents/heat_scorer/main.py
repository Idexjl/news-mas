import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.common import agent_registry
from src.common.agent_bootstrap import bootstrap_agent
from src.common.observability import configure_langsmith, get_logger, get_tracer, setup_telemetry
from src.common.schemas import HeatScorerInput, HeatScorerOutput
from src.auth.workload_identity import get_identity_provider
# [DPOP-TODO] Replace SecurityHeaderMiddleware with DPoPAuthMiddleware once
# Entra ID app registrations are provisioned. See DPOP_IMPLEMENTATION_GUIDE.md §4 (Phase 5).
# from src.common.auth.middleware import DPoPAuthMiddleware
# from src.common.auth.agent_identity import load_agent_identity
from .agent import run_agent

setup_telemetry("heat-scorer")
configure_langsmith()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Heat Scorer Agent")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_identity_provider = get_identity_provider()


class SecurityHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in ("/health",):
            try:
                _identity_provider.validate_incoming_token(
                    request.headers.get("X-MAS-Secret", "")
                )
            except ValueError as exc:
                return JSONResponse(status_code=401, content={"detail": str(exc)})
        return await call_next(request)


app.add_middleware(SecurityHeaderMiddleware)
# [DPOP-TODO] Remove SecurityHeaderMiddleware above and activate DPoP:
#   _identity = load_agent_identity("heat_scorer")
#   app.add_middleware(DPoPAuthMiddleware,
#       tenant_id=_identity.tenant_id,
#       audience=f"api://{_identity.client_id}",
#   )


@app.on_event("startup")
async def startup() -> None:
    tracer = get_tracer("heat-scorer")
    with tracer.start_as_current_span("agent.bootstrap") as span:
        registry_url = os.getenv("REGISTRY_URL", "http://localhost:8000")
        agent_id = os.getenv("AGENT_ID", "heat-scorer")
        try:
            config = await bootstrap_agent(registry_url, _identity_provider, agent_id)
            app.state.agent_config = config
            span.set_attribute("agent.id", config.agent_id)
            span.set_attribute("agent.model_provider", config.model_provider)
            span.set_attribute("agent.model_id", config.model_id)
            agent_registry.register(
                config.agent_id,
                capabilities=config.capabilities,
                endpoint=os.getenv("SERVICE_URL", "http://localhost:8002"),
            )
            logger.info("agent_bootstrapped", extra={
                "agent_id": config.agent_id,
                "model_provider": config.model_provider,
                "network_tier": config.network_tier,
            })
        except Exception as exc:
            logger.warning("bootstrap_failed", extra={"error": str(exc)})
            app.state.agent_config = None


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "heat_scorer"}


@app.post("/run", response_model=HeatScorerOutput)
@limiter.limit("30/minute")
async def run(request: Request, payload: HeatScorerInput):
    logger.info("run requested", extra={"run_id": payload.run_id})
    config = getattr(request.app.state, "agent_config", None)
    model_id = config.model_id if config else None
    model_provider = config.model_provider if config else None
    return await run_agent(payload, model_id=model_id, model_provider=model_provider)
