"""
Relevance Gate Agent — final pass/tombstone decision for each topic.

Uses Claude Haiku by default; MODEL_OVERRIDE applies (unlike summarizer/reviewer).
Fast paths skip the LLM for obvious cases — tokens and latency saved on clear calls.
LLM path handles borderline heat/confidence combinations.

Decision priority (evaluated in order):
  1. budget_exhausted     → fast tombstone (deterministic)
  2. heat < 0.2           → fast tombstone (too_cold)
  3. LOW conf + heat < 0.4 → fast tombstone (low_confidence)
  4. heat >= 0.8 + HIGH   → fast pass
  5. reviewer pass + heat >= 0.7 → fast pass
  6. reviewer fail        → fast tombstone (review_failed)
  7. everything else      → LLM decision

LLM unavailable → DEGRADED: default to pass rather than incorrectly excluding.

OTel spans emitted:
  relevance_gate.evaluate  (parent) → topic_id, run_id, decision, gate_path,
                                       heat_score, confidence, retry_count,
                                       budget_exhausted
  relevance_gate.llm_call  (child)  → model_id, prompt_version, attempt
  relevance_gate.tombstone (event)  → topic_id, reason_code
                                       (never tombstone_message — user content)

Security invariants:
  - topic_text NEVER in spans, logs, or error messages
  - tombstone_message NEVER in spans
  - reviewer_last_feedback NEVER in spans or logs
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional

from dotenv import load_dotenv

try:
    from langsmith import traceable as _ls_traceable
except ImportError:
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec

try:
    import ollama as _ollama
    _OllamaResponseError = _ollama.ResponseError
except ImportError:
    _ollama = None  # type: ignore[assignment]
    _OllamaResponseError = Exception  # type: ignore[assignment,misc]

try:
    import anthropic as _anthropic
    _AnthropicAPIError = _anthropic.APIError
    _AnthropicRateLimitError = _anthropic.RateLimitError
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _AnthropicAPIError = Exception  # type: ignore[assignment,misc]
    _AnthropicRateLimitError = Exception  # type: ignore[assignment,misc]

from src.auth.token_service import validate_token
from src.common.agent_bootstrap import resolve_model
from src.common.error_codes import (
    AUTH_FAILED,
    GATE_LLM_UNAVAILABLE,
    INVALID_LLM_RESPONSE,
)
from src.common.observability import (
    configure_langsmith,
    get_logger,
    get_tracer,
    log_pipeline_error,
    record_span_error,
    setup_telemetry,
)
from src.common.pipeline_errors import AgentResult, ErrorSeverity, PipelineError  # noqa: F401
from src.common.prompt_loader import get_system_prompt, load_prompt, make_run_config
from src.common.schemas import (
    DigestTombstone,
    RelevanceGateInput,
    RelevanceGateOutput,
)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_PROVIDER = "anthropic"
_AGENT_ID = "relevance-gate"
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "gate.relevance"
_MAX_JSON_RETRIES = 2

logger = get_logger(__name__)


# ── AAP token validation ───────────────────────────────────────────────────────

def _validate_aap_token(token: str) -> dict[str, Any]:
    secret = os.getenv("MAS_SECRET_KEY", "")
    claims = validate_token(token, secret_key=secret)
    caps: list[str] = claims.get("aap_capabilities", [])
    if _REQUIRED_CAPABILITY not in caps:
        raise PermissionError(
            f"AAP token missing required capability '{_REQUIRED_CAPABILITY}'. "
            f"Granted: {caps}"
        )
    agent_info = claims.get("aap_agent", {})
    logger.info(
        "aap_token_validated",
        extra={
            "actor_agent": agent_info.get("agent_name"),
            "task_id": claims.get("aap_task", {}).get("task_id"),
            "delegation_depth": claims.get("aap_delegation", {}).get("depth"),
        },
    )
    return claims


# ── Data quality normalisation ────────────────────────────────────────────────

def _normalise_input(inp: RelevanceGateInput) -> RelevanceGateInput:
    """Clamp/default any out-of-range or missing fields to safe values."""
    updates: dict[str, Any] = {}
    if not (0.0 <= inp.heat_score <= 1.0):
        logger.warning(
            "gate.heat_score_out_of_range",
            extra={"run_id": inp.run_id, "heat_score": inp.heat_score},
        )
        updates["heat_score"] = 0.0
    if inp.overall_confidence not in ("HIGH", "MEDIUM", "LOW"):
        logger.warning("gate.invalid_confidence", extra={"run_id": inp.run_id})
        updates["overall_confidence"] = "LOW"
    if inp.reviewer_verdict not in ("pass", "fail"):
        logger.warning("gate.invalid_verdict", extra={"run_id": inp.run_id})
        updates["reviewer_verdict"] = "fail"
    return inp.model_copy(update=updates) if updates else inp


# ── Tombstone helpers ─────────────────────────────────────────────────────────

_DEFAULT_TOMBSTONE_MESSAGES: dict[str, str] = {
    "budget_exhausted": "Coverage for this topic was not completed within the refresh window.",
    "too_cold":         "Limited news activity for this topic this week.",
    "low_confidence":   "Limited reliable sources were available for this topic.",
    "review_failed":    "Coverage of this topic did not meet quality standards.",
    "gate_decision":    "Nothing significant to report for this topic this week.",
}


def _make_tombstone(
    inp: RelevanceGateInput,
    reason_code: str,
    tombstone_message: Optional[str] = None,
) -> DigestTombstone:
    message = tombstone_message or _DEFAULT_TOMBSTONE_MESSAGES.get(
        reason_code, "This topic was not included this week."
    )
    return DigestTombstone(
        topic_id=inp.topic_id,
        topic_display_name=inp.topic_text,   # reader's own query — safe to show
        tombstone_message=message,
        heat_score=inp.heat_score,
        confidence=inp.overall_confidence,  # type: ignore[arg-type]
        reason_code=reason_code,            # type: ignore[arg-type]
    )


# ── Fast path evaluation ──────────────────────────────────────────────────────

def _fast_path(
    inp: RelevanceGateInput,
) -> tuple[Optional[str], str, Optional[str]]:
    """
    Evaluate deterministic gate paths — no LLM required.

    Returns (decision, gate_path, reason_code):
      decision   "pass" | "tombstone" | None (proceed to LLM)
      gate_path  "fast_pass" | "fast_tombstone" | "llm_decision"
      reason_code  set on tombstone, None otherwise
    """
    # Priority 1 — budget exhausted: always tombstone
    if inp.budget_exhausted:
        logger.info("gate.budget_exhausted", extra={"run_id": inp.run_id, "topic_id": inp.topic_id})
        return "tombstone", "fast_tombstone", "budget_exhausted"

    # Priority 2 — too cold: not enough activity regardless of confidence
    if inp.heat_score < 0.2:
        return "tombstone", "fast_tombstone", "too_cold"

    # Priority 3 — low confidence at low heat: unreliable AND quiet
    if inp.overall_confidence == "LOW" and inp.heat_score < 0.4:
        return "tombstone", "fast_tombstone", "low_confidence"

    # Priority 4 — high heat + high confidence: always include
    if inp.heat_score >= 0.8 and inp.overall_confidence == "HIGH":
        return "pass", "fast_pass", None

    # Priority 5 — reviewer already approved at good heat: trust the reviewer
    if inp.reviewer_verdict == "pass" and inp.heat_score >= 0.7:
        return "pass", "fast_pass", None

    # Priority 6 — reviewer failed outside normal flow: tombstone conservatively
    if inp.reviewer_verdict == "fail":
        return "tombstone", "fast_tombstone", "review_failed"

    # No fast path — borderline case requires LLM judgement
    return None, "llm_decision", None


# ── LLM backends ─────────────────────────────────────────────────────────────

def _get_ollama_client() -> Any:
    if _ollama is None:
        raise RuntimeError("ollama package is not installed")
    return _ollama.AsyncClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


@_ls_traceable(
    run_type="llm",
    name="relevance-gate-ollama",
    tags=["relevance-gate", "gemma4", "local"],
    metadata={"model": "gemma4:e4b"},
)
async def _call_ollama(
    prompt: str,
    system: str,
    run_id: str,
    *,
    model_id: str,
) -> dict[str, Any]:
    """LangSmith-traced Ollama inference. Returns Ollama-native response dict."""
    client = _get_ollama_client()
    response = await client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    return dict(response)


@_ls_traceable(
    run_type="llm",
    name="relevance-gate-anthropic",
    tags=["relevance-gate", "anthropic", "cloud"],
    metadata={"provider": "anthropic"},
)
async def _call_anthropic(
    prompt: str,
    system: str,
    run_id: str,
    *,
    model_id: str,
) -> dict[str, Any]:
    """LangSmith-traced Anthropic inference. Returns Ollama-compatible shape."""
    if _anthropic is None:
        raise RuntimeError("anthropic package is not installed")
    async with _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) as client:
        message = await client.messages.create(
            model=model_id,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    text = message.content[0].text if message.content else ""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return {"message": {"content": text.strip()}}


# ── LangSmith-traced LLM gate decision ────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="relevance_gate",
    tags=["agent:relevance_gate"],
)
async def _traced_llm_decision(
    inp: RelevanceGateInput,
    *,
    model_id: str,
    provider: str,
    tracer: Any,
) -> tuple[str, Optional[str]]:
    """
    LangSmith-traced gate decision via LLM.

    Only called on the LLM path — fast paths never invoke this function,
    so no LangSmith trace is created for obvious pass/tombstone cases.

    Returns (decision, tombstone_message). Raises RuntimeError on unrecoverable
    failure so the caller can apply the DEGRADED-PASS fallback.
    """
    prompt_data = load_prompt("relevance_gate", _PROMPT_VERSION)
    system_prompt = get_system_prompt("relevance_gate", _PROMPT_VERSION)
    user_template: str = prompt_data.get("user_template", "")

    feedback_block = ""
    if inp.reviewer_last_feedback:
        feedback_block = f"Reviewer feedback: {inp.reviewer_last_feedback}\n\n"

    user_prompt = user_template.format(
        heat_score=f"{inp.heat_score:.2f}",
        overall_confidence=inp.overall_confidence,
        reviewer_verdict=inp.reviewer_verdict,
        retry_count=inp.retry_count,
        feedback_block=feedback_block,
    )

    parsed: dict[str, Any] | None = None

    with tracer.start_as_current_span("relevance_gate.llm_call") as llm_span:
        llm_span.set_attribute("model_id", model_id)
        llm_span.set_attribute("prompt_version", _PROMPT_VERSION)

        for attempt in range(1 + _MAX_JSON_RETRIES):
            llm_span.set_attribute("attempt", attempt)

            if provider == "anthropic":
                resp = await _call_anthropic(user_prompt, system_prompt, inp.run_id, model_id=model_id)
            else:
                resp = await _call_ollama(user_prompt, system_prompt, inp.run_id, model_id=model_id)

            raw = resp.get("message", {}).get("content", "")
            try:
                candidate = json.loads(raw)
                if "decision" not in candidate:
                    raise ValueError("Missing required key: decision")
                parsed = candidate
                break
            except (json.JSONDecodeError, ValueError):
                if attempt < _MAX_JSON_RETRIES:
                    logger.warning(
                        "gate.json_parse_failed_retrying",
                        extra={"run_id": inp.run_id, "attempt": attempt},
                    )
                else:
                    err = PipelineError(
                        error_code=INVALID_LLM_RESPONSE,
                        severity=ErrorSeverity.RETRYABLE,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message="Gate LLM returned unparseable JSON after retries",
                        retry_hint="Retry; reinforce JSON-only output in system prompt",
                        context={"attempts": _MAX_JSON_RETRIES + 1},
                    )
                    record_span_error(llm_span, err)
                    raise RuntimeError(err.message)

    decision_raw = (parsed.get("decision") or "").lower()  # type: ignore[union-attr]
    decision: str = "pass" if decision_raw == "pass" else "tombstone"
    tombstone_message: Optional[str] = parsed.get("tombstone_message") or None  # type: ignore[union-attr]
    reasoning: Optional[str] = parsed.get("reasoning") or None  # type: ignore[union-attr]

    # Never log tombstone_message — it may be derived from topic content
    logger.info(
        "gate.llm_raw_decision",
        extra={"run_id": inp.run_id, "decision": decision},
    )

    return decision, tombstone_message


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: RelevanceGateInput,
    *,
    tracer: Any = None,
    model_id: str | None = None,
    model_provider: str | None = None,
) -> RelevanceGateOutput:
    """
    Public entry point called by main.py and the Phase 2 orchestrator.

    MODEL_OVERRIDE applies — unlike summarizer/reviewer, the gate can be
    switched to a cheaper model for development without quality impact.
    Default: anthropic / claude-haiku-4-5-20251001.

    LLM unavailable → DEGRADED: defaults to pass so uncertain topics reach
    the digest rather than being silently excluded.
    """
    load_dotenv()
    setup_telemetry("news-mas-relevance-gate")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("relevance-gate")

    provider, effective_model = resolve_model(
        model_id or _DEFAULT_MODEL,
        model_provider or _DEFAULT_PROVIDER,
        _AGENT_ID,
    )

    make_run_config(
        "relevance_gate",
        _PROMPT_VERSION,
        run_id=inp.run_id,
        topic_id=inp.topic_id or None,
    )

    logger.info(
        "gate_start",
        extra={
            "run_id": inp.run_id,
            "topic_id": inp.topic_id,
            "heat_score": inp.heat_score,
            "overall_confidence": inp.overall_confidence,
            "reviewer_verdict": inp.reviewer_verdict,
            "retry_count": inp.retry_count,
            "budget_exhausted": inp.budget_exhausted,
            "model_id": effective_model,
            "langsmith_active": langsmith_active,
            # Never log topic_text, tombstone_reason, reviewer_last_feedback
        },
    )

    # ── AAP token validation ───────────────────────────────────────────────────
    if inp.aap_token:
        try:
            _validate_aap_token(inp.aap_token)
        except PermissionError as exc:
            auth_error = PipelineError(
                error_code=AUTH_FAILED,
                severity=ErrorSeverity.FATAL,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                message=str(exc),
                context={},
            )
            log_pipeline_error(logger, auth_error)
            return RelevanceGateOutput(run_id=inp.run_id, success=False, error=auth_error)

    # ── Data quality normalisation ─────────────────────────────────────────────
    norm = _normalise_input(inp)

    # ── Fast path evaluation ───────────────────────────────────────────────────
    fast_decision, gate_path, reason_code = _fast_path(norm)

    with active_tracer.start_as_current_span("relevance_gate.evaluate") as eval_span:
        eval_span.set_attribute("topic_id", norm.topic_id)
        eval_span.set_attribute("run_id", norm.run_id)
        eval_span.set_attribute("gate_path", gate_path)
        eval_span.set_attribute("heat_score", norm.heat_score)
        eval_span.set_attribute("confidence", norm.overall_confidence)
        eval_span.set_attribute("retry_count", norm.retry_count)
        eval_span.set_attribute("budget_exhausted", norm.budget_exhausted)
        # Never record topic_text, tombstone_message, or reviewer_last_feedback

        degraded_error: PipelineError | None = None
        decision: str
        tombstone_message: Optional[str] = None

        if fast_decision is not None:
            # ── Fast path ─────────────────────────────────────────────────────
            decision = fast_decision
            if decision == "tombstone":
                if reason_code == "budget_exhausted" and norm.tombstone_reason:
                    tombstone_message = norm.tombstone_reason
                else:
                    tombstone_message = _DEFAULT_TOMBSTONE_MESSAGES.get(reason_code or "")
            logger.info(
                "gate.fast_path",
                extra={
                    "run_id": norm.run_id,
                    "topic_id": norm.topic_id,
                    "gate_path": gate_path,
                    "decision": decision,
                    "reason_code": reason_code,
                },
            )
        else:
            # ── LLM path ──────────────────────────────────────────────────────
            reason_code = "gate_decision"
            decision = "pass"

            try:
                decision, tombstone_message = await _traced_llm_decision(
                    norm,
                    model_id=effective_model,
                    provider=provider,
                    tracer=active_tracer,
                )
            except Exception as exc:
                logger.warning(
                    "gate.llm_unavailable.defaulting_pass",
                    extra={
                        "run_id": norm.run_id,
                        "topic_id": norm.topic_id,
                        "error_type": type(exc).__name__,
                        # Never log exc message — may contain content
                    },
                )
                decision = "pass"
                reason_code = None
                tombstone_message = None
                degraded_error = PipelineError(
                    error_code=GATE_LLM_UNAVAILABLE,
                    severity=ErrorSeverity.DEGRADED,
                    agent_id=_AGENT_ID,
                    run_id=norm.run_id,
                    message="Gate LLM unavailable; defaulting to pass",
                    retry_hint="Check LLM backend connectivity and model availability",
                    context={
                        "error_type": type(exc).__name__,
                        "model_id": effective_model,
                    },
                )
                log_pipeline_error(logger, degraded_error)
                record_span_error(eval_span, degraded_error)

        # ── Finalise OTel span ─────────────────────────────────────────────────
        eval_span.set_attribute("decision", decision)

        if decision == "tombstone" and reason_code:
            # Never include tombstone_message in span — it is user-facing content
            eval_span.add_event(
                "relevance_gate.tombstone",
                attributes={
                    "topic_id": norm.topic_id,
                    "reason_code": reason_code,
                },
            )

    # ── Build and return output ────────────────────────────────────────────────
    warnings: list[str] = ["gate_llm_unavailable"] if degraded_error else []

    if decision == "tombstone" and reason_code:
        tombstone = _make_tombstone(norm, reason_code, tombstone_message)
        logger.info(
            "gate_complete",
            extra={
                "run_id": norm.run_id,
                "topic_id": norm.topic_id,
                "decision": "tombstone",
                "gate_path": gate_path,
                "reason_code": reason_code,
                # Never log tombstone.tombstone_message
            },
        )
        return RelevanceGateOutput(
            run_id=norm.run_id,
            success=True,
            decision="tombstone",
            gate_path=gate_path,  # type: ignore[arg-type]
            tombstone=tombstone,
            error=degraded_error,
            warnings=warnings,
        )

    logger.info(
        "gate_complete",
        extra={
            "run_id": norm.run_id,
            "topic_id": norm.topic_id,
            "decision": "pass",
            "gate_path": gate_path,
        },
    )
    return RelevanceGateOutput(
        run_id=norm.run_id,
        success=True,
        decision="pass",
        gate_path=gate_path,  # type: ignore[arg-type]
        tombstone=None,
        error=degraded_error,
        warnings=warnings,
    )
