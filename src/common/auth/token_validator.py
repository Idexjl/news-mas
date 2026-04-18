"""
Access-token validation with DPoP cnf/jkt claim binding for Entra ID tokens.

Entra ID JWKS endpoint (fetch public keys for signature verification):
  https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys

Entra DPoP reference:
  https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


# [DPOP-TODO] Add to pyproject.toml: `PyJWT[crypto]>=2.8` or `python-jose[cryptography]`
# and `httpx` (already present) for async JWKS fetch.


def compute_jwk_thumbprint(jwk: dict[str, Any]) -> str:
    """
    Compute the JWK thumbprint (RFC 7638) for a public key.

    The thumbprint is used to verify the cnf/jkt binding in Entra-issued
    DPoP tokens. The cnf claim contains:
        {"cnf": {"jkt": "<base64url(sha256(canonical-JWK)>)"}}

    Canonical member sets (sorted, no whitespace):
      EC P-256: {crv, kty, x, y}
      RSA:      {e, kty, n}

    [DPOP-TODO] Implement:
        EC_MEMBERS = ["crv", "kty", "x", "y"]
        RSA_MEMBERS = ["e", "kty", "n"]
        members = EC_MEMBERS if jwk.get("kty") == "EC" else RSA_MEMBERS
        canonical = json.dumps(
            {k: jwk[k] for k in members},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(canonical).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    """
    raise NotImplementedError("[DPOP-TODO] compute_jwk_thumbprint not implemented")


async def fetch_entra_jwks(tenant_id: str) -> list[dict[str, Any]]:
    """
    Fetch and return Entra ID public keys from the JWKS endpoint.

    [DPOP-TODO] Implement with caching:
      - Cache JWKS response for 1 hour (Entra rotates keys infrequently)
      - On kid-mismatch during verification, force-refresh once before failing
        (handles key rotation without downtime)
      - Use httpx.AsyncClient (already in dependencies)

        async with httpx.AsyncClient() as client:
            url = (
                f"https://login.microsoftonline.com/{tenant_id}"
                "/discovery/v2.0/keys"
            )
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()["keys"]
    """
    raise NotImplementedError("[DPOP-TODO] fetch_entra_jwks not implemented")


async def validate_entra_token(
    access_token: str,
    dpop_proof: str,
    *,
    tenant_id: str,
    audience: str,
    required_scopes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate an Entra ID DPoP-bound access token and its proof.

    Full validation chain (all steps must pass):

      1. Decode token header to get `kid`
      2. Fetch JWKS from Entra (cached) and find matching key by `kid`
      3. Verify JWT signature using the matching public key
      4. Verify standard claims:
           iss  == https://sts.windows.net/{tenant_id}/ (v1) or
                   https://login.microsoftonline.com/{tenant_id}/v2.0 (v2)
           aud  == *audience* (the receiving agent's app ID URI)
           exp  > now
           nbf  <= now
      5. Verify token_type claim == "DPoP" (Entra sets this on DPoP-bound tokens)
      6. Extract cnf/jkt claim and compare to JWK thumbprint from DPoP proof:
           expected_jkt = compute_jwk_thumbprint(proof_header["jwk"])
           assert token_claims["cnf"]["jkt"] == expected_jkt
      7. Call dpop.verify_dpop_proof() to validate htm/htu/iat/jti/ath

    [DPOP-TODO] cnf/jkt binding is the core security property — step 6 ensures
    the token can only be used by the agent holding the matching private key.
    A stolen token without the private key is useless.

    [DPOP-TODO] For required_scopes, check the `scp` (delegated) or `roles`
    (application) claim depending on the flow type (OBO vs client credentials).
    Client credentials flows use `roles`; OBO flows use `scp`.

    Returns validated claims dict on success.
    Raises ValueError with a safe (non-leaking) message on failure.
    """
    raise NotImplementedError("[DPOP-TODO] validate_entra_token not implemented")
