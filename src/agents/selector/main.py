from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.common.observability import get_logger, setup_telemetry
from src.common.schemas import SelectorInput, SelectorOutput
from src.auth.workload_identity import get_identity_provider
# [DPOP-TODO] Replace SecurityHeaderMiddleware with DPoPAuthMiddleware once
# Entra ID app registrations are provisioned. See DPOP_IMPLEMENTATION_GUIDE.md §4 (Phase 5).
# from src.common.auth.middleware import DPoPAuthMiddleware
# from src.common.auth.agent_identity import load_agent_identity
from .agent import run_agent

setup_telemetry("selector")
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Selector Agent")
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
#   _identity = load_agent_identity("selector")
#   app.add_middleware(DPoPAuthMiddleware,
#       tenant_id=_identity.tenant_id,
#       audience=f"api://{_identity.client_id}",
#   )
# Entra ref: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "selector"}


@app.post("/run", response_model=SelectorOutput)
@limiter.limit("30/minute")
async def run(request: Request, payload: SelectorInput):
    logger.info("run requested", extra={"run_id": payload.run_id})
    return await run_agent(payload)
