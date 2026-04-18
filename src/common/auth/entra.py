"""
Microsoft Entra ID token acquisition for agent-to-agent (A2A) flows.

Two grant types are used in news-mas:
  1. client_credentials — orchestrator → agent and agent → agent (no user context)
  2. On-Behalf-Of (OBO) — user-delegated runs where the delegation chain must be preserved

Entra token endpoint:
  POST https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token

Entra DPoP reference:
  https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenCache:
    """
    In-memory token cache keyed by (tenant_id, client_id, scope, flow).

    DPoP access tokens ARE cached (they have a normal expiry, typically 1h).
    DPoP *proofs* are NEVER cached — generate a fresh proof per request.
    """
    _store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._store.get(key)
        # 30-second buffer avoids using a token that expires mid-request
        if entry and entry["expires_at"] > time.time() + 30:
            return entry
        return None

    def set(self, key: str, token_response: dict[str, Any]) -> None:
        expires_in = token_response.get("expires_in", 3600)
        self._store[key] = {**token_response, "expires_at": time.time() + expires_in}

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


_token_cache = TokenCache()


async def get_client_credentials_token(
    *,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    scope: str,
    dpop_private_key: Any | None = None,
) -> dict[str, Any]:
    """
    Acquire an Entra ID token via client credentials flow (RFC 6749 §4.4).

    Used for machine-to-machine calls with no user context — the typical pattern
    for scheduled digest runs and all orchestrator → agent calls.

    When *dpop_private_key* is provided the request is DPoP-bound:
      - Generates a DPoP proof for the token endpoint URI
      - Adds `token_type=DPoP` to the POST body
      - Sets the `DPoP:` header on the token request
      - Entra returns `token_type: "DPoP"` in the response

    Scope convention for A2A calls:
      "api://{target_agent_client_id}/.default"
    See DPOP_IMPLEMENTATION_GUIDE.md §3.5 for app registration setup.

    [DPOP-TODO] Implement with httpx.AsyncClient:

        cache_key = f"cc:{tenant_id}:{client_id}:{scope}"
        cached = _token_cache.get(cache_key)
        if cached:
            return cached

        token_url = (
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        )
        body = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
        }
        headers = {}
        if dpop_private_key is not None:
            from src.common.auth.dpop import generate_dpop_proof
            body["token_type"] = "DPoP"
            headers["DPoP"] = generate_dpop_proof(
                method="POST",
                uri=token_url,
                private_key=dpop_private_key,
            )

        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=body, headers=headers)
            resp.raise_for_status()
            token_response = resp.json()

        _token_cache.set(cache_key, token_response)
        return token_response

    [DPOP-TODO] For production, prefer certificate-based client_assertion over
    client_secret. See DPOP_IMPLEMENTATION_GUIDE.md §3.1 (app registrations).
    """
    raise NotImplementedError(
        "[DPOP-TODO] get_client_credentials_token not implemented"
    )


async def get_obo_token(
    *,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    user_assertion: str,
    scope: str,
    dpop_private_key: Any | None = None,
) -> dict[str, Any]:
    """
    Acquire a downstream token via Entra On-Behalf-Of flow.

    Used when a user-initiated request propagates through multiple agents and
    each hop must preserve the user's identity for audit and policy enforcement.

    Entra OBO uses grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer,
    which differs from RFC 8693 token exchange. Key divergences:

      RFC 8693 grant_type   : urn:ietf:params:oauth:grant-type:token-exchange
      Entra OBO grant_type  : urn:ietf:params:oauth:grant-type:jwt-bearer

      RFC 8693 subject_token_type : required
      Entra OBO                   : omit (Entra infers jwt type)

      RFC 8693 requested_token_type : optional
      Entra OBO                     : omit (Entra always returns access_token)

    The returned token is bound to THIS agent's DPoP key, not the upstream
    caller's key — each hop rebinds to its own key pair.

    See DPOP_IMPLEMENTATION_GUIDE.md §3.3 for the full divergence analysis
    and guidance on OBO vs client_credentials per scenario.

    [DPOP-TODO] Implement with httpx.AsyncClient:

        token_url = (
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        )
        body = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "client_id": client_id,
            "client_secret": client_secret,
            "assertion": user_assertion,
            "scope": scope,
            "requested_token_use": "on_behalf_of",
        }
        headers = {}
        if dpop_private_key is not None:
            from src.common.auth.dpop import generate_dpop_proof
            body["token_type"] = "DPoP"
            headers["DPoP"] = generate_dpop_proof(
                method="POST",
                uri=token_url,
                private_key=dpop_private_key,
            )
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    [DPOP-TODO] OBO tokens should NOT be long-cached — the upstream user
    assertion has its own expiry and the user's access may be revoked.
    Cache with a shorter TTL or skip caching for OBO tokens.
    """
    raise NotImplementedError("[DPOP-TODO] get_obo_token not implemented")
