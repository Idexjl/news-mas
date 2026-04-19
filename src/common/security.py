from __future__ import annotations

import hmac
import logging
import os
import re
import unicodedata
from typing import Any, Final, Protocol

# ── Injection / attack detection patterns ────────────────────────────────────

_RAW_PATTERNS: Final[list[str]] = [
    # Prompt injection
    r"ignore\b.{0,40}\binstructions\b",
    r"disregard\b.{0,20}\binstructions\b",
    r"\bdo\s+not\s+follow\b",
    r"bypass\b.{0,20}(instructions|guidelines|rules)\b",
    r"you\s+are\s+now\b",
    r"disregard\s+(your|all|previous)",
    r"\bsystem\s*:",
    r"<\s*system\s*>",
    r"override\s+(your|all)?\s*(instructions?|guidelines?|rules?)",
    r"forget\s+(everything|all|your)",
    r"new\s+instructions?\s*:",
    r"\[\[system\]\]",
    r"act\s+as\s+(if\s+you\s+(are|were)|an?\s+\w)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"dan\s+mode",
    # SQL injection
    r"union\s+select\b",
    r"drop\s+table\b",
    r"insert\s+into\b",
    r"delete\s+from\b",
    r"exec\s*\(",
    r"xp_cmdshell",
    # XSS / script injection
    r"<\s*script[\s>]",
    r"javascript\s*:",
    r"vbscript\s*:",
    r"on\w+\s*=\s*[\"']",
    # Path traversal
    r"\.\.[/\\]",
    r"/etc/passwd",
    r"\\windows\\system32",
]

_COMPILED: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _RAW_PATTERNS
]


def normalize_input(text: str) -> str:
    """Apply NFKC normalisation to collapse homoglyph / fullwidth variants."""
    return unicodedata.normalize("NFKC", text)


def detect_injection(text: str) -> list[str]:
    """
    Return the list of raw pattern strings that matched *text*.
    An empty list means no threats detected.

    [SIEM-TODO] In production, every non-empty return value from this function
    is a Sentinel-relevant security event. The calling agent logs
    "egress.security.injection" or "injection_detected_discarding_result" —
    both feed into Sentinel via the OTel log pipeline and can trigger an
    automated incident response Playbook:
      1. Analytics Rule: alert when injection_detected_total increases in any
         5-minute window (maps to the EgressInjectionDetected Prometheus alert).
      2. Playbook (Logic App): on alert, call the Sentinel API to create an
         Incident, enrich it with the egress.host and pattern_count from the
         span attributes (never the matched content — HIPAA constraint), and
         notify the SOC via Teams/email.
      3. Playbook can also invoke the circuit-breaker: set run status to
         "killed" in FernetStorage so downstream agents refuse work even if
         their tokens are still valid (see is_run_active() below).
      Pattern count and egress host are the only fields safe to include in
      Sentinel incident descriptions — never the matched content or raw text.
    """
    normalised = normalize_input(text)
    return [
        _RAW_PATTERNS[i]
        for i, pattern in enumerate(_COMPILED)
        if pattern.search(normalised)
    ]


def is_safe_input(text: str) -> bool:
    """Return True only when no injection patterns are detected."""
    return len(detect_injection(text)) == 0


# ── Run-level circuit breaker ─────────────────────────────────────────────────

# Statuses that mean the orchestrator has halted this run.
# Any agent that receives a valid token for one of these runs must refuse the
# work — this is the primary operational control for the Entra introspection gap.
# See DPOP_IMPLEMENTATION_GUIDE.md §3.6.
_INACTIVE_STATUSES: Final[frozenset[str]] = frozenset({"cancelled", "killed", "interrupted"})


class _RunStore(Protocol):
    """Structural protocol — any object with a load(collection, id) method qualifies."""

    def load(self, collection: str, id: str) -> dict[str, Any]: ...  # noqa: A002


def is_run_active(
    run_id: str,
    *,
    storage: _RunStore,
    aap_claims: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Return True when *run_id* is active and work should proceed.
    Return False when the run has been cancelled, killed, or interrupted.

    A run not yet present in storage is treated as active (fail-open), which
    allows agents to process requests before the orchestrator has persisted the
    initial run state.

    When the run is inactive but the caller presented a valid auth token, the
    situation is a security event: a token outlived the work it was issued for.
    A WARNING is logged with the full *aap_claims* dict for forensic triage.
    """
    _log = logger or logging.getLogger(__name__)
    try:
        run = storage.load("runs", run_id)
    except KeyError:
        return True  # run not yet persisted — let it through

    status: str = str(run.get("status", "")).lower()
    if status in _INACTIVE_STATUSES:
        _log.warning(
            "circuit_breaker_triggered: run inactive but token still valid",
            extra={
                "run_id": run_id,
                "run_status": status,
                "aap_claims": aap_claims,
                "security_event": True,
            },
        )
        return False
    return True


# [DPOP-TODO] validate_shared_secret is a development-only authentication
# mechanism. Replace with DPoP + Entra ID token validation in production.
# Production path: src/common/auth/middleware.py → DPoPAuthMiddleware
# See DPOP_IMPLEMENTATION_GUIDE.md §4 (Phase 5 — middleware rollout).
# Entra ref: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
#
# [SIEM-TODO] Auth failures from validate_shared_secret are high-value Sentinel
# signals for Entra ID Identity Protection. In production (once DPoP is active):
#   - Every validation failure becomes an Entra "risky sign-in" event, since
#     each agent has its own Managed Identity. Sentinel's Identity Protection
#     Analytics Rule fires when the same agent_id fails validation repeatedly.
#   - Sentinel UEBA builds an authentication baseline per agent (expected source
#     IP, request cadence, capability set). A deviation — e.g. an agent
#     authenticating from an unexpected source subnet — generates a medium-risk
#     entity alert even if the credential itself is valid.
#   - In the interim (shared-secret era), instrument the ValueError raise below
#     with a structured log event: {"event": "auth_failure", "source_ip": ...}
#     so Sentinel can correlate auth failures with other signals in the same
#     time window (injection detections, anomalous egress, registry PUT attempts).
def validate_shared_secret(provided: str) -> None:
    """
    Compare *provided* against the MAS_SECRET_KEY env var using a
    constant-time comparison to prevent timing attacks.

    Raises ValueError on mismatch.
    No-op when MAS_SECRET_KEY is not set (dev convenience).
    """
    expected = os.getenv("MAS_SECRET_KEY", "")
    if not expected:
        return
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise ValueError("Invalid or missing shared secret")
