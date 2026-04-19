"""
AAP (Agent Authorization Protocol) token service.

Mints, exchanges, and validates HS256 JWTs carrying the seven AAP claim
sections that thread task context, oversight requirements, and delegation
history through a multi-agent call chain.

Signing uses stdlib hmac + hashlib — no external JWT library required.

[DPOP-TODO] Replace HS256 shared-secret signing with DPoP-bound tokens
once the Entra ID / DPoP layer (src/common/auth/) is fully wired in:
  - mint_token   : inject cnf claim (JWK thumbprint of agent's DPoP key)
  - exchange_token: validate inbound DPoP proof, bind outbound token to
                    receiving agent's key via cnf/jkt
  - validate_token: verify cnf/jkt against DPoP proof on every call
See DPOP_IMPLEMENTATION_GUIDE.md §4 for the full migration checklist.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any

MAX_DEPTH: int = 3

_REQUIRED_AAP_SECTIONS: frozenset[str] = frozenset({
    "aap_agent",
    "aap_task",
    "aap_capabilities",
    "aap_oversight",
    "aap_delegation",
    "aap_context",
    "aap_audit",
})


# ── low-level JWT helpers (stdlib only) ───────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(segment: str) -> bytes:
    pad = 4 - len(segment) % 4
    if pad != 4:
        segment += "=" * pad
    return base64.urlsafe_b64decode(segment)


def _sign(signing_input: bytes, secret: str) -> str:
    raw = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return _b64url(raw)


def _encode_jwt(payload: dict[str, Any], secret: str) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    return f"{header}.{body}.{_sign(signing_input, secret)}"


def _decode_and_verify(token: str, secret: str) -> dict[str, Any]:
    """Verify signature and expiry, return payload dict."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token: expected three dot-separated segments")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = _sign(signing_input, secret)

    # Constant-time comparison guards against timing side-channels.
    if not hmac.compare_digest(expected, sig_b64):
        raise ValueError("Invalid token signature: verification failed")

    payload: dict[str, Any] = json.loads(_b64url_decode(payload_b64))

    if payload.get("exp", 0) < int(time.time()):
        raise ValueError("Token has expired")

    return payload


def _assert_aap_sections(claims: dict[str, Any]) -> None:
    missing = _REQUIRED_AAP_SECTIONS - claims.keys()
    if missing:
        # Sort for a stable, readable error message.
        raise ValueError(f"Missing required AAP sections: {sorted(missing)}")


# ── public API ─────────────────────────────────────────────────────────────────

def mint_token(
    *,
    subject: str,
    agent_name: str,
    model_provider: str,
    model_id: str,
    capabilities: list[str],
    task_id: str,
    task_description: str,
    data_sensitivity: str,
    oversight_required: bool,
    oversight_level: str,
    context: dict[str, Any],
    secret_key: str,
    ttl_seconds: int = 3600,
    max_depth: int = MAX_DEPTH,
) -> str:
    """
    Mint a new AAP token for *agent_name* operating on *task_id*.

    [DPOP-TODO] Once cnf claim support lands, accept a ``dpop_public_jwk``
    parameter and embed it as ``cnf: {jwk: ...}`` so the token is
    cryptographically bound to the minting agent's DPoP key from birth.
    See DPOP_IMPLEMENTATION_GUIDE.md §4.3.

    [SIEM-TODO] The AAP token payload — especially aap_agent, aap_task,
    aap_delegation, and aap_context — provides the full identity and task
    context that Sentinel needs to enrich every downstream security event.
    In production, log a structured mint event (token_id, agent_name,
    task_id, data_sensitivity, delegation depth, run_id from context) so
    Sentinel can correlate all activity in a pipeline run back to the
    original mint event. The token_id from aap_audit becomes the correlation
    key across Sentinel incidents — if an injection is detected downstream,
    Sentinel can look up the token_id to find which orchestrator triggered
    the run, what topic was being searched, and which agent chain was active.
    data_sensitivity is especially important: Sentinel Analytics Rules can
    apply stricter alerting thresholds when data_sensitivity == "PHI".
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + ttl_seconds,
        "aap_agent": {
            "agent_name": agent_name,
            "model_provider": model_provider,
            "model_id": model_id,
        },
        "aap_task": {
            "task_id": task_id,
            "task_description": task_description,
            "data_sensitivity": data_sensitivity,
        },
        "aap_capabilities": list(capabilities),
        "aap_oversight": {
            "oversight_required": oversight_required,
            "oversight_level": oversight_level,
        },
        "aap_delegation": {
            "depth": 0,
            "chain": [{"agent_name": agent_name}],
            "max_depth": max_depth,
        },
        "aap_context": dict(context),
        "aap_audit": {
            "token_id": str(uuid.uuid4()),
            "minted_at": now,
        },
    }
    return _encode_jwt(payload, secret_key)


def exchange_token(
    *,
    token: str,
    new_agent_name: str,
    new_model_provider: str,
    new_model_id: str,
    new_capabilities: list[str],
    secret_key: str,
    ttl_seconds: int = 3600,
) -> str:
    """
    Issue a delegated AAP token for *new_agent_name* derived from *token*.

    Preserved from original: sub, aap_task (incl. data_sensitivity),
    aap_oversight, aap_delegation.max_depth.
    Updated: aap_agent, aap_capabilities, aap_delegation.depth/chain.

    Raises ValueError when the current delegation depth equals max_depth —
    the chain cannot be extended further.

    [DPOP-TODO] Validate the inbound DPoP proof bound to *token* before
    accepting it (call dpop.verify_dpop_proof with the DPoP header from the
    HTTP request). After validation, bind the outbound token to the
    receiving agent's key via cnf/jkt. See DPOP_IMPLEMENTATION_GUIDE.md §4.2.
    """
    claims = validate_token(token, secret_key=secret_key)

    delegation = claims["aap_delegation"]
    current_depth: int = delegation["depth"]
    max_depth_val: int = delegation["max_depth"]

    if current_depth >= max_depth_val:
        raise ValueError(
            f"Delegation depth limit reached: depth {current_depth} "
            f"equals max_depth {max_depth_val} — cannot delegate further"
        )

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": claims["sub"],
        "iat": now,
        "exp": now + ttl_seconds,
        "aap_agent": {
            "agent_name": new_agent_name,
            "model_provider": new_model_provider,
            "model_id": new_model_id,
        },
        "aap_task": claims["aap_task"],
        "aap_capabilities": list(new_capabilities),
        "aap_oversight": claims["aap_oversight"],
        "aap_delegation": {
            "depth": current_depth + 1,
            "chain": list(delegation["chain"]) + [{"agent_name": new_agent_name}],
            "max_depth": max_depth_val,
        },
        "aap_context": claims.get("aap_context", {}),
        "aap_audit": {
            "token_id": str(uuid.uuid4()),
            "minted_at": now,
        },
    }
    return _encode_jwt(payload, secret_key)


def validate_token(token: str, *, secret_key: str) -> dict[str, Any]:
    """
    Verify *token* and return its decoded claims.

    Raises ValueError for: expired token, wrong/missing signature, or any
    of the seven required AAP sections being absent.

    [DPOP-TODO] Add a ``dpop_proof`` parameter and verify cnf/jkt binding
    once DPoP is active end-to-end. Until then the shared-secret HS256
    check is the sole authentication mechanism.
    See DPOP_IMPLEMENTATION_GUIDE.md §4.1.

    [SIEM-TODO] Failed validations here are among the highest-value signals
    in the entire system for Sentinel:
      - "Invalid token signature" → possible token forgery or key rotation
        issue. If seen in bursts, may indicate a supply-chain compromise of
        the shared secret. Sentinel alert: ≥3 signature failures in 5 minutes
        from the same source IP → high-severity incident.
      - "Token has expired" → may indicate a replay attempt with a captured
        token, or a clock-skew issue between services (low severity alone, but
        high severity if combined with an injection detection in the same run).
      - "Missing required AAP sections" → structurally malformed token,
        possibly from a non-standard client probing the API.
      In production, wrap _decode_and_verify in a try/except and emit a
      structured log event for each failure type:
        {"event": "aap_token_validation_failure", "reason": exc.__class__.__name__,
         "token_id": extracted_token_id_if_parseable, "agent_id": ...}
      Sentinel can then join these events with network flows (from egress spans)
      to detect patterns that neither signal alone would surface.
    """
    claims = _decode_and_verify(token, secret_key)
    _assert_aap_sections(claims)
    return claims
