"""
DPoP (Demonstrating Proof-of-Possession) proof generation and verification.
RFC 9449: https://datatracker.ietf.org/doc/html/rfc9449

Microsoft Entra ID requires a DPoP proof on EVERY request, not just token
issuance. See DPOP_IMPLEMENTATION_GUIDE.md §2 and Entra DPoP reference:
https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
"""
from __future__ import annotations

import hashlib
import base64
import time
import uuid
from typing import Any


# [DPOP-TODO] Install the `cryptography` package (add to pyproject.toml
# dependencies) and import key primitives:
#
#   from cryptography.hazmat.primitives.asymmetric import ec
#   from cryptography.hazmat.primitives.asymmetric.ec import (
#       EllipticCurvePrivateKey, EllipticCurvePublicKey, SECP256R1,
#   )
#   from cryptography.hazmat.primitives import serialization
#   from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
#
# Also add `PyJWT>=2.8` or `python-jose[cryptography]` for JWT signing.


def generate_dpop_keypair() -> tuple[Any, Any]:
    """
    Generate an EC P-256 key pair for use in DPoP proofs.

    Entra ID accepts both EC (P-256 / ES256) and RSA (RS256, PS256) DPoP keys.
    EC P-256 is recommended — shorter proofs, faster verification.

    [DPOP-TODO] Implement:
        from cryptography.hazmat.primitives.asymmetric import ec
        private_key = ec.generate_private_key(ec.SECP256R1())
        return private_key, private_key.public_key()

    Key storage guidance:
      - Azure-hosted agents: generate once at startup, store private key in
        Azure Key Vault (load via Managed Identity). Key Vault handles rotation.
      - Cross-cloud agents (AWS/GCP): inject as a mounted secret or environment
        variable (PEM-encoded). See DPOP_IMPLEMENTATION_GUIDE.md §3.4.
      - Never log, serialise to disk, or include in telemetry.
    """
    raise NotImplementedError("[DPOP-TODO] generate_dpop_keypair not implemented")


def private_key_to_jwk(private_key: Any) -> dict[str, Any]:
    """
    Export the PUBLIC component of *private_key* as a JWK dict.

    The JWK is embedded in the DPoP proof JOSE header (RFC 9449 §4.2).
    Only the public key is ever included — never export private key material.

    [DPOP-TODO] For EC P-256:
        pub = private_key.public_key()
        pub_numbers = pub.public_key().public_numbers()  # type: ignore[union-attr]
        x = base64.urlsafe_b64encode(
            pub_numbers.x.to_bytes(32, "big")
        ).rstrip(b"=").decode()
        y = base64.urlsafe_b64encode(
            pub_numbers.y.to_bytes(32, "big")
        ).rstrip(b"=").decode()
        return {"kty": "EC", "crv": "P-256", "x": x, "y": y}
    """
    raise NotImplementedError("[DPOP-TODO] private_key_to_jwk not implemented")


def generate_dpop_proof(
    *,
    method: str,
    uri: str,
    private_key: Any,
    access_token: str | None = None,
) -> str:
    """
    Generate a single-use DPoP proof JWT for one HTTP request (RFC 9449 §4.2).

    JOSE header:
        {"typ": "dpop+jwt", "alg": "ES256", "jwk": {<public key JWK>}}

    Payload claims:
        htm  — HTTP method, e.g. "POST"
        htu  — HTTP URI without query string or fragment
        iat  — issued-at (seconds since epoch)
        jti  — unique identifier (UUID4 recommended)
        ath  — base64url(sha256(access_token))  ← only when AT is present

    Entra ID requires a fresh proof on EVERY protected request — the proof is
    bound to a specific method + URI and is single-use. Never reuse a proof.

    [DPOP-TODO] Implement using PyJWT:
        import jwt as pyjwt

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": private_key_to_jwk(private_key),
        }
        payload: dict[str, Any] = {
            "htm": method.upper(),
            "htu": uri.split("?")[0].split("#")[0],
            "iat": int(time.time()),
            "jti": str(uuid.uuid4()),
        }
        if access_token is not None:
            digest = hashlib.sha256(access_token.encode()).digest()
            payload["ath"] = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

        return pyjwt.encode(payload, private_key, algorithm="ES256",
                            headers=header)
    """
    raise NotImplementedError("[DPOP-TODO] generate_dpop_proof not implemented")


def verify_dpop_proof(
    *,
    proof_jwt: str,
    method: str,
    uri: str,
    access_token: str | None = None,
    max_age_seconds: int = 60,
) -> dict[str, Any]:
    """
    Verify a DPoP proof received on an inbound request (RFC 9449 §4.3).

    Validation steps (all must pass):
      1. JOSE header typ == "dpop+jwt"
      2. Signature valid against the public key in the jwk header
      3. htm matches *method* (case-insensitive)
      4. htu matches *uri* (scheme + host + path; ignore query/fragment)
      5. iat is within *max_age_seconds* of now (clock-skew defence)
      6. jti not seen before within the validity window (replay prevention)
      7. ath == base64url(sha256(*access_token*)) when access_token provided

    Returns the decoded payload dict on success.
    Raises ValueError with a non-leaking message on any failure.

    [DPOP-TODO] Step 6 requires a jti store. Options:
      - In-process: dict with TTL-based eviction (acceptable for single-instance)
      - Redis: required for horizontally-scaled agents
      Key: jti value. TTL: max_age_seconds + clock-skew buffer (e.g. +30s).

    [DPOP-TODO] Entra ID sets iat validity to ~60s. Align max_age_seconds with
    your Entra tenant's configured DPoP nonce window if using nonces (Entra
    supports nonce-based replay prevention as an alternative to jti tracking).
    """
    raise NotImplementedError("[DPOP-TODO] verify_dpop_proof not implemented")
