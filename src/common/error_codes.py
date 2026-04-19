"""
Pipeline error code constants.

Used as the error_code field on PipelineError. Keep codes uppercase and
underscore-separated. Add new codes here before raising them — the orchestrator
maps these codes to routing decisions (retry, skip, abort, degrade).
"""

# ── Search / retrieval ────────────────────────────────────────────────────────
NO_RESULTS = "NO_RESULTS"
INSUFFICIENT_RESULTS = "INSUFFICIENT_RESULTS"

# ── Filtering / scoring ───────────────────────────────────────────────────────
ALL_FILTERED = "ALL_FILTERED"
HEAT_SCORE_TOO_LOW = "HEAT_SCORE_TOO_LOW"
NO_VIABLE_CANDIDATES = "NO_VIABLE_CANDIDATES"

# ── Summarisation quality ─────────────────────────────────────────────────────
SUMMARY_TOO_SHORT = "SUMMARY_TOO_SHORT"
SUMMARY_TOO_LONG = "SUMMARY_TOO_LONG"
CITATION_INVALID = "CITATION_INVALID"
CONSTRAINT_VIOLATED = "CONSTRAINT_VIOLATED"

# ── Resource / budget ─────────────────────────────────────────────────────────
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

# ── Security ──────────────────────────────────────────────────────────────────
INJECTION_DETECTED = "INJECTION_DETECTED"
PHI_DETECTED = "PHI_DETECTED"

# ── Infrastructure ────────────────────────────────────────────────────────────
REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
AUTH_FAILED = "AUTH_FAILED"
