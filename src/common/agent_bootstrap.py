"""
Agent bootstrap: fetches AgentConfig from the registry at agent startup.

Call bootstrap_agent() once in the FastAPI startup event. The returned
AgentConfig is the single source of truth for the agent's model, capabilities,
token budget, prompt version, and network tier.

[WORKLOAD-IDENTITY-TODO] In production, replace the MAS_SECRET_KEY header
with a Managed Identity token obtained from identity_provider.get_identity_token().
The registry will validate that token against the platform identity service
(Entra ID JWKS endpoint) instead of comparing against the shared secret.
See src/auth/workload_identity.py [WORKLOAD-IDENTITY-TODO] for the full guide.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

from src.auth.workload_identity import WorkloadIdentityProvider
from src.common.observability import get_logger
from src.registry.models import AgentConfig

logger = get_logger(__name__)


async def bootstrap_agent(
    registry_url: str,
    identity_provider: WorkloadIdentityProvider,
    agent_id: str | None = None,
    _client: Optional[httpx.AsyncClient] = None,
) -> AgentConfig:
    """
    Obtain this agent's configuration from the registry.

    Steps:
      1. Resolve agent_id from parameter → AGENT_ID env var → provider claims.
      2. Present identity proof to the registry GET endpoint.
      3. Deserialise and return the AgentConfig.

    Raises ValueError if the registry returns 404 (agent not seeded) or any
    non-2xx response. Raises httpx.HTTPError on connectivity failures.

    [WORKLOAD-IDENTITY-TODO] identity_provider.get_identity_token() returns a
    platform-signed JWT in production (Azure IMDS / AWS STS / GCP metadata).
    The registry validates that JWT against Entra JWKS — no shared secret
    involved. Until DPoP is active, we fall back to MAS_SECRET_KEY because
    LocalDevIdentityProvider.validate_incoming_token() compares against the
    raw secret, not a JWT.
    """
    claims = identity_provider.get_workload_claims()
    resolved_id = (
        agent_id
        or os.getenv("AGENT_ID")
        or claims.get("agent_id")
        or "unknown"
    )

    # [WORKLOAD-IDENTITY-TODO] Replace this line with:
    #   identity_token = identity_provider.get_identity_token()
    # once the registry validates Managed Identity JWTs.
    identity_token = os.getenv("MAS_SECRET_KEY", "dev-secret")

    logger.info("bootstrap_starting", extra={"agent_id": resolved_id, "registry_url": registry_url})

    async def _fetch(c: httpx.AsyncClient) -> httpx.Response:
        return await c.get(
            f"{registry_url}/agents/{resolved_id}/config",
            headers={"X-MAS-Secret": identity_token},
            timeout=10.0,
        )

    if _client is not None:
        response = await _fetch(_client)
    else:
        async with httpx.AsyncClient() as client:
            response = await _fetch(client)

    if response.status_code == 404:
        raise ValueError(
            f"Registry has no config for agent '{resolved_id}'. "
            "Ensure the registry server is running and seeded. "
            "Set AGENT_ID to the correct agent identifier."
        )

    if response.status_code == 401:
        raise ValueError(
            f"Registry rejected identity for agent '{resolved_id}'. "
            "Check MAS_SECRET_KEY matches the registry's secret."
        )

    response.raise_for_status()
    config = AgentConfig(**response.json())
    logger.info(
        "bootstrap_complete",
        extra={
            "agent_id": config.agent_id,
            "model_provider": config.model_provider,
            "model_id": config.model_id,
            "network_tier": config.network_tier,
        },
    )
    return config
