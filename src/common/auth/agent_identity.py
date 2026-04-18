"""
Per-agent Entra ID client registration loader.

Each news-mas agent has its own Entra app registration. Shared credentials
across agents would break per-agent token revocation and audit trails.
See DPOP_IMPLEMENTATION_GUIDE.md §3.1 for the full registration guide.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentIdentity:
    """
    Entra ID client registration for one agent instance.

    Attributes:
      tenant_id      — Entra tenant GUID (shared across all agents)
      client_id      — app registration Application (client) ID
      client_secret  — secret value; prefer certificate in production (see below)
      target_scopes  — scopes this agent is authorised to request from others
      dpop_key       — loaded private key for DPoP proof generation (None = DPoP off)

    [DPOP-TODO] Replace `client_secret: str` with certificate support.
    In production, store a certificate in Azure Key Vault and authenticate with:
        client_assertion = build_client_assertion_jwt(cert, client_id, tenant_id)
    This avoids putting secret values in environment variables and enables
    Key Vault-managed rotation without redeployment.

    [DPOP-TODO] Replace `dpop_key: Any` with the concrete private key type once
    the cryptography package is added:
        from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
        dpop_key: PrivateKeyTypes | None = None
    """
    tenant_id: str
    client_id: str
    client_secret: str
    target_scopes: list[str] = field(default_factory=list)
    dpop_key: Any | None = None  # [DPOP-TODO] type as PrivateKeyTypes once cryptography is added


def load_agent_identity(agent_name: str) -> AgentIdentity:
    """
    Load the Entra ID client registration for *agent_name* from environment.

    Expected environment variables:
      ENTRA_TENANT_ID                  — shared across all agents
      AGENT_{UPPER_NAME}_CLIENT_ID     — app registration client ID
      AGENT_{UPPER_NAME}_CLIENT_SECRET — secret (or cert ref; see below)
      AGENT_{UPPER_NAME}_SCOPES        — comma-separated target scopes

    Example for the heat_scorer agent:
      ENTRA_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
      AGENT_HEAT_SCORER_CLIENT_ID=yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy
      AGENT_HEAT_SCORER_CLIENT_SECRET=<secret>
      AGENT_HEAT_SCORER_SCOPES=api://<phase1_judge_client_id>/.default

    [DPOP-TODO] Load the DPoP private key after building the identity object.
    For Azure-hosted agents, load from Key Vault via Managed Identity:
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential
        kv_url = os.environ.get("AZURE_KEY_VAULT_URL", "")
        if kv_url:
            kv_client = SecretClient(vault_url=kv_url,
                                     credential=DefaultAzureCredential())
            pem = kv_client.get_secret(f"dpop-key-{agent_name}").value
            identity.dpop_key = load_private_key_from_pem(pem)

    For cross-cloud agents (AWS/GCP), load from a mounted secret:
        pem_path = os.environ.get(f"AGENT_{prefix}_DPOP_KEY_PATH", "")
        if pem_path:
            with open(pem_path, "rb") as f:
                identity.dpop_key = load_private_key_from_pem(f.read())
    See DPOP_IMPLEMENTATION_GUIDE.md §3.4.

    [DPOP-TODO] Fail fast at startup if tenant_id or client_id are missing.
    Agents without valid credentials must not accept traffic — add:
        if not identity.tenant_id or not identity.client_id:
            raise RuntimeError(
                f"Agent '{agent_name}' is missing Entra credentials. "
                "Set ENTRA_TENANT_ID and AGENT_{NAME}_CLIENT_ID."
            )
    """
    prefix = agent_name.upper()
    tenant_id = os.environ.get("ENTRA_TENANT_ID", "")
    client_id = os.environ.get(f"AGENT_{prefix}_CLIENT_ID", "")
    client_secret = os.environ.get(f"AGENT_{prefix}_CLIENT_SECRET", "")
    scopes_raw = os.environ.get(f"AGENT_{prefix}_SCOPES", "")
    target_scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()]

    return AgentIdentity(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        target_scopes=target_scopes,
        dpop_key=None,  # [DPOP-TODO] load key here (see docstring above)
    )
