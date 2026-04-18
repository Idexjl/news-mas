from __future__ import annotations

import hmac
import os
import re
import unicodedata
from typing import Final

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


# [DPOP-TODO] validate_shared_secret is a development-only authentication
# mechanism. Replace with DPoP + Entra ID token validation in production.
# Production path: src/common/auth/middleware.py → DPoPAuthMiddleware
# See DPOP_IMPLEMENTATION_GUIDE.md §4 (Phase 5 — middleware rollout).
# Entra ref: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
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
