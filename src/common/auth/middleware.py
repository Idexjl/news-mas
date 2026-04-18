"""
DPoP authentication middleware for FastAPI agents.

Replaces SecurityHeaderMiddleware (shared-secret, dev-only) once Entra ID
app registrations and DPoP keys are provisioned for all agents.

See DPOP_IMPLEMENTATION_GUIDE.md §4 (Phase 5 — middleware rollout) for the
step-by-step transition plan.

Entra DPoP reference:
  https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.common.observability import get_logger

logger = get_logger(__name__)

_OPEN_PATHS = frozenset({"/health"})


class DPoPAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates DPoP-bound Entra ID access tokens on every protected request.

    Expected inbound headers:
      Authorization: DPoP {access_token}
      DPoP: {dpop_proof_jwt}

    Full validation chain (delegated to auth sub-modules):
      1. Parse "DPoP <token>" from Authorization header
      2. Extract DPoP proof from DPoP header
      3. Call token_validator.validate_entra_token() which:
           a. Fetches/caches Entra JWKS and verifies token signature
           b. Validates iss, aud, exp, nbf, token_type == "DPoP"
           c. Extracts cnf/jkt and verifies it matches the DPoP proof's jwk
           d. Calls dpop.verify_dpop_proof() for htm/htu/iat/jti/ath checks
      4. Stores validated claims in request.state.token_claims for handlers

    [DPOP-TODO] Complete dispatch() implementation once dpop.py and
    token_validator.py are fully implemented. The stub below performs
    structural header checks only and falls through to legacy auth during
    the transition period.

    [DPOP-TODO] For inter-agent calls the orchestrator must generate a FRESH
    DPoP proof per outbound request. The proof is bound to method + URI and
    is single-use. See DPOP_IMPLEMENTATION_GUIDE.md §2.3 (proof lifecycle).

    [DPOP-TODO] Configure tenant_id and audience per agent via
    load_agent_identity(). Each agent's audience is its own App ID URI:
      audience = f"api://{identity.client_id}"
    """

    def __init__(self, app, *, tenant_id: str = "", audience: str = "") -> None:
        super().__init__(app)
        self.tenant_id = tenant_id
        self.audience = audience
        # [DPOP-TODO] Warm the JWKS cache here so the first real request is not
        # slowed by a cold fetch:
        #   asyncio.create_task(fetch_entra_jwks(tenant_id))

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        dpop_header = request.headers.get("DPoP", "")

        # Structural check — both headers must be present for DPoP auth
        has_dpop_auth = auth_header.startswith("DPoP ") and bool(dpop_header)

        if has_dpop_auth:
            # [DPOP-TODO] Replace this block with real validation:
            #
            #   from src.common.auth.token_validator import validate_entra_token
            #   access_token = auth_header[len("DPoP "):]
            #   try:
            #       claims = await validate_entra_token(
            #           access_token=access_token,
            #           dpop_proof=dpop_header,
            #           tenant_id=self.tenant_id,
            #           audience=self.audience,
            #       )
            #       request.state.token_claims = claims
            #   except ValueError as exc:
            #       logger.warning("DPoP validation failed",
            #                      extra={"reason": str(exc)})
            #       return JSONResponse(
            #           status_code=401,
            #           content={"detail": "Unauthorized"},
            #       )
            #
            # Once implemented, remove the fall-through below and this comment.
            logger.warning(
                "DPoP headers present but validation not yet implemented — "
                "falling through. See [DPOP-TODO] in middleware.py."
            )
            return await call_next(request)

        # Transition period: fall through to SecurityHeaderMiddleware (which
        # runs after this one in the middleware stack) so agents remain
        # functional before full DPoP rollout.
        # [DPOP-TODO] Change this branch to a hard 401 once all callers send
        # DPoP tokens and SecurityHeaderMiddleware is removed.
        logger.warning("DPoP headers absent — falling through to legacy auth")
        return await call_next(request)
