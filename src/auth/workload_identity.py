"""
Workload identity providers for agent-to-agent authentication.

Each provider abstracts the platform-specific mechanism for obtaining an
identity token (proof of who this agent is) and exchanging it for an AAP
token (what the agent is authorised to do in the current task).

Current implementation: LocalDevIdentityProvider only (shared-secret, dev/test).
Production providers are stubbed below with [WORKLOAD-IDENTITY-TODO] guides.

[WORKLOAD-IDENTITY-TODO] AzureManagedIdentityProvider:
  1. Call IMDS at http://169.254.169.254/metadata/identity/oauth2/token
     (header: Metadata: true, query: api-version=2018-02-01, resource=<audience>)
     to obtain a managed-identity access token for the running VM/container.
  2. Present that token to Entra ID's workload-identity-federation endpoint to
     exchange it for a service-principal access token scoped to the target
     agent's API audience (client-credentials flow, federated credential assertion).
  3. Pass the resulting Entra token as Authorization: Bearer in /run calls, then
     mint an AAP token via exchange_for_aap_token for pipeline authorisation.
  Azure IMDS ref:
    https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service
  Federated credential ref:
    https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation

[WORKLOAD-IDENTITY-TODO] AWSSTSFederationProvider:
  1. Call AWS STS AssumeRoleWithWebIdentity to obtain an OIDC identity token for
     the running IAM role (sts.amazonaws.com GetCallerIdentity + OIDC endpoint).
  2. Present that OIDC token to Entra via a configured federation trust
     (requires Entra app registration with an AWS-issued token trust and a
     matching IAM role policy that permits STS:AssumeRoleWithWebIdentity).
  3. Exchange the resulting Entra token for an AAP token.
  AWS STS ref:
    https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
  AWS → Entra federation ref:
    https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation-create-trust

[WORKLOAD-IDENTITY-TODO] GCPWorkloadIdentityProvider:
  1. Call the GCP metadata server at
     http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity
     (header: Metadata-Flavor: Google, query: audience=<entra-app-id-uri>)
     to obtain an OIDC identity token for the running service account.
  2. Present the GCP OIDC token to Entra via OIDC federation
     (requires Entra app registration with a GCP-issued token trust; the
     audience in the GCP token must match the Entra Application ID URI).
  3. Exchange the resulting Entra token for an AAP token.
  GCP metadata server ref:
    https://cloud.google.com/compute/docs/metadata/overview
  GCP → Entra OIDC federation ref:
    https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation-create-trust

[WORKLOAD-IDENTITY-TODO] GitHubActionsProvider:
  1. Read ACTIONS_ID_TOKEN_REQUEST_URL and ACTIONS_ID_TOKEN_REQUEST_TOKEN
     from the GitHub Actions environment.
  2. POST to ACTIONS_ID_TOKEN_REQUEST_URL with the bearer token to obtain a
     GitHub-issued OIDC JWT for the running workflow.
  3. Present that JWT to Entra via a federation trust configured on the app
     registration (subject filter: repo:org/repo:environment:production or
     repo:org/repo:ref:refs/heads/main — match your deployment rules).
  4. Exchange the resulting deployment-scoped Entra token for an AAP token.
  GitHub OIDC ref:
    https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/about-security-hardening-with-openid-connect
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from abc import ABC, abstractmethod

from src.auth.token_service import mint_token


# ── Identity JWT helpers (stdlib-only, for LocalDevIdentityProvider) ──────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(segment: str) -> bytes:
    pad = 4 - len(segment) % 4
    if pad != 4:
        segment += "=" * pad
    return base64.urlsafe_b64decode(segment)


def _encode_identity_jwt(payload: dict, secret: str) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    sig = _b64url(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def _decode_identity_jwt(token: str, secret: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid identity token: expected three dot-separated segments")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = _b64url(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig_b64):
        raise ValueError("Identity token signature verification failed")
    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("exp", 0) < int(time.time()):
        raise ValueError("Identity token has expired")
    return payload


# ── Abstract base ─────────────────────────────────────────────────────────────

class WorkloadIdentityProvider(ABC):
    """
    Abstract interface for workload identity.

    Separates two concerns:
      - Outbound identity: how this agent proves who it is to the platform and
        to other agents (get_identity_token, exchange_for_aap_token).
      - Inbound validation: how this agent validates callers on incoming requests
        (validate_incoming_token).
    """

    @abstractmethod
    def get_identity_token(self) -> str:
        """
        Obtain a platform-issued identity token for this agent.

        In production this calls the platform metadata endpoint
        (IMDS / STS OIDC / GCP metadata server / GitHub OIDC).
        Returns a signed JWT asserting this workload's identity.
        """

    @abstractmethod
    def exchange_for_aap_token(self, token: str, capabilities: list[str]) -> str:
        """
        Exchange a platform identity token for a pipeline AAP token.

        In production: validates the platform token against Entra ID, then
        mints an AAP token bound to the verified identity and the given
        capabilities.
        """

    @abstractmethod
    def get_workload_claims(self) -> dict:
        """
        Return identifying metadata for this workload instance.

        Returned keys: agent_id, environment, region, provider_type.
        """

    @abstractmethod
    def validate_incoming_token(self, token: str) -> None:
        """
        Validate a token presented by a caller on an inbound /run request.

        Raises ValueError on validation failure.
        In production: validates the Entra JWT signature and claims via JWKS.
        For local dev: validates the shared secret via constant-time comparison.
        """


# ── Local development implementation ─────────────────────────────────────────

class LocalDevIdentityProvider(WorkloadIdentityProvider):
    """
    # WARNING: NOT FOR PRODUCTION USE.
    #
    # Uses MAS_SECRET_KEY (shared HMAC secret) for local development and testing.
    # Provides no platform attestation and no cryptographic binding to a specific
    # workload — any holder of the secret key can forge an identity token.
    #
    # [WORKLOAD-IDENTITY-TODO] Replace with AzureManagedIdentityProvider in Phase 2
    # of the DPoP rollout once Entra app registrations are provisioned per agent.
    # See DPOP_IMPLEMENTATION_GUIDE.md §3.1 and §4, and the module-level
    # AzureManagedIdentityProvider guide in this file.
    """

    def __init__(
        self,
        secret_key: str | None = None,
        agent_id: str = "local-dev-agent",
    ) -> None:
        # Use None as the explicit "read from env" sentinel so that
        # passing secret_key="" is treated as "no secret, no-op validation".
        self._secret_key: str = (
            os.getenv("MAS_SECRET_KEY", "dev-secret") if secret_key is None else secret_key
        )
        self._agent_id: str = agent_id
        self._environment: str = os.getenv("ENVIRONMENT", "development")

    def get_identity_token(self) -> str:
        now = int(time.time())
        payload = {
            "sub": self._agent_id,
            "iat": now,
            "exp": now + 300,  # 5-minute identity assertion
            "provider_type": "local",
            "environment": self._environment,
        }
        return _encode_identity_jwt(payload, self._secret_key)

    def exchange_for_aap_token(self, token: str, capabilities: list[str]) -> str:
        claims = _decode_identity_jwt(token, self._secret_key)
        return mint_token(
            subject=claims.get("sub", self._agent_id),
            agent_name=self._agent_id,
            model_provider="local",
            model_id="local",
            capabilities=capabilities,
            task_id=str(uuid.uuid4()),
            task_description="local-dev-task",
            data_sensitivity="internal",
            oversight_required=False,
            oversight_level="none",
            context={"environment": self._environment},
            secret_key=self._secret_key,
        )

    def get_workload_claims(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "environment": self._environment,
            "region": "local",
            "provider_type": "local",
        }

    def validate_incoming_token(self, token: str) -> None:
        if not self._secret_key:
            return
        if not hmac.compare_digest(token.encode(), self._secret_key.encode()):
            raise ValueError("Invalid or missing shared secret")


# ── Config and factory ────────────────────────────────────────────────────────

_VALID_PROVIDERS: frozenset[str] = frozenset(
    {"local", "azure-managed-identity", "aws-sts", "gcp-workload"}
)


class WorkloadIdentityConfig:
    """
    Reads WORKLOAD_IDENTITY_PROVIDER from the environment and returns the
    matching provider via get_provider().

    Valid values:
      local                  — LocalDevIdentityProvider (dev/test only)
      azure-managed-identity — AzureManagedIdentityProvider (Azure-hosted agents)
      aws-sts                — AWSSTSFederationProvider (AWS-hosted agents)
      gcp-workload           — GCPWorkloadIdentityProvider (GCP-hosted agents)
    """

    def __init__(self) -> None:
        self.provider_type: str = os.getenv(
            "WORKLOAD_IDENTITY_PROVIDER", "local"
        ).strip()

    def get_provider(self) -> WorkloadIdentityProvider:
        return _build_provider(self.provider_type)


def get_identity_provider() -> WorkloadIdentityProvider:
    """Module-level factory. Reads WORKLOAD_IDENTITY_PROVIDER from the environment."""
    return WorkloadIdentityConfig().get_provider()


def _build_provider(provider_type: str) -> WorkloadIdentityProvider:
    if provider_type == "local":
        return LocalDevIdentityProvider()

    if provider_type == "azure-managed-identity":
        # [WORKLOAD-IDENTITY-TODO] Implement AzureManagedIdentityProvider.
        # Call IMDS at 169.254.169.254, exchange via Entra federation trust.
        # See AzureManagedIdentityProvider guide in this module's docstring.
        raise NotImplementedError(
            "AzureManagedIdentityProvider is not yet implemented. "
            "See [WORKLOAD-IDENTITY-TODO] AzureManagedIdentityProvider in "
            "src/auth/workload_identity.py for the implementation guide."
        )

    if provider_type == "aws-sts":
        # [WORKLOAD-IDENTITY-TODO] Implement AWSSTSFederationProvider.
        # Call STS OIDC endpoint, present token to Entra configured federation trust.
        # See AWSSTSFederationProvider guide in this module's docstring.
        raise NotImplementedError(
            "AWSSTSFederationProvider is not yet implemented. "
            "See [WORKLOAD-IDENTITY-TODO] AWSSTSFederationProvider in "
            "src/auth/workload_identity.py for the implementation guide."
        )

    if provider_type == "gcp-workload":
        # [WORKLOAD-IDENTITY-TODO] Implement GCPWorkloadIdentityProvider.
        # Call GCP metadata server, exchange with Entra via OIDC federation.
        # See GCPWorkloadIdentityProvider guide in this module's docstring.
        raise NotImplementedError(
            "GCPWorkloadIdentityProvider is not yet implemented. "
            "See [WORKLOAD-IDENTITY-TODO] GCPWorkloadIdentityProvider in "
            "src/auth/workload_identity.py for the implementation guide."
        )

    raise ValueError(
        f"Unknown workload identity provider '{provider_type}'. "
        f"Valid values: {sorted(_VALID_PROVIDERS)}. "
        "Set WORKLOAD_IDENTITY_PROVIDER in .env — see .env.example for options."
    )
