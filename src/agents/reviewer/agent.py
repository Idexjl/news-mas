"""
Reviewer Agent — independent LLM-as-judge for summarizer output.

Model: claude-sonnet-4-6 (hardcoded; MODEL_OVERRIDE never applies here).
Evaluates three dimensions: citation accuracy, constraint compliance, content substance.
Part of the Phase 2 retry loop — returns pass/fail with structured feedback.

OTel spans emitted:
  reviewer.review    (parent) → topic_id, run_id, verdict, retry_count,
                                 issue_count, model_id
  reviewer.llm_call  (child)  → model_id, prompt_version, token_estimate,
                                 retry_count
  reviewer.phi_check (child)  → phi_detected (bool)
  reviewer.fail      (event)  → issue_types[] (never descriptions or content)

Security invariants:
  - Summary content NEVER in spans, logs, or error messages
  - Issue descriptions NEVER in spans
  - Source content NEVER in spans or logs
  - Topic/constraint text NEVER in spans or logs
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

from dotenv import load_dotenv

try:
    from langsmith import traceable as _ls_traceable
except ImportError:
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec

try:
    import anthropic as _anthropic
    _AnthropicRateLimitError = _anthropic.RateLimitError
    _AnthropicAPIError = _anthropic.APIError
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _AnthropicRateLimitError = Exception  # type: ignore[assignment,misc]
    _AnthropicAPIError = Exception  # type: ignore[assignment,misc]

from src.auth.token_service import validate_token
from src.common.error_codes import (
    AUTH_FAILED,
    INVALID_LLM_RESPONSE,
    PHI_DETECTED,
    REVIEWER_UNAVAILABLE,
)
from src.common.observability import (
    configure_langsmith,
    get_logger,
    get_tracer,
    log_pipeline_error,
    record_span_error,
    setup_telemetry,
)
from src.common.pii_scrubber import detect_phi_in_output
from src.common.pipeline_errors import AgentResult, ErrorSeverity, PipelineError  # noqa: F401
from src.common.prompt_loader import get_system_prompt, load_prompt, make_run_config
from src.common.schemas import (
    Citation,
    ReviewIssue,
    ReviewerInput,
    ReviewerOutput,
    SearchResult,
)

_AGENT_ID = "reviewer"
_MODEL_ID = "claude-sonnet-4-6"   # hardcoded; never overridden by MODEL_OVERRIDE
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "review.content"
_MAX_API_RETRIES = 3
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


# ── Prompt formatting helpers ──────────────────────────────────────────────────

def _format_key_points(key_points: list[str]) -> str:
    if not key_points:
        return "  (none)"
    return "\n".join(f"  - {kp}" for kp in key_points)


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "  (no citations provided)"
    parts: list[str] = []
    for c in citations:
        parts.append(
            f"  [{c.index}] {c.url}\n"
            f"      Title: {c.title}\n"
            f"      Quote: \"{c.quote}\""
        )
    return "\n".join(parts)


def _format_sources(sources: list[SearchResult]) -> str:
    if not sources:
        return "  (no source articles provided — citation accuracy check skipped)"
    parts: list[str] = []
    for idx, s in enumerate(sources, start=1):
        content = (s.content or "").strip()[:3000]  # cap per source to stay within budget
        parts.append(
            f"  [{idx}] {s.url}\n"
            f"      Title: {s.title}\n"
            f"      Content: {content}\n"
            f"      {'─' * 60}"
        )
    return "\n\n".join(parts)


def _format_constraints(constraints: list[str]) -> str:
    if not constraints:
        return "  (none — accept any content)"
    return "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(constraints))


def _build_retry_context(retry_count: int) -> str:
    if retry_count == 0:
        return ""
    return (
        f"RETRY CONTEXT: This is attempt {retry_count}. Apply reasonable standards — "
        f"do not fail for minor issues that were not present in previous feedback.\n\n"
    )


# ── Anthropic LLM call ─────────────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="reviewer-claude-sonnet",
    tags=["reviewer", "anthropic", "claude-sonnet"],
    metadata={"model": _MODEL_ID},
)
async def _call_claude(
    user_prompt: str,
    system_prompt: str,
    *,
    run_id: str,
    max_tokens: int = 1024,
) -> str:
    """LangSmith-traced Claude Sonnet call. Retries up to _MAX_API_RETRIES for transient errors."""
    if _anthropic is None:
        raise RuntimeError("anthropic package is not installed")

    import re

    last_exc: Exception | None = None
    for attempt in range(_MAX_API_RETRIES + 1):
        try:
            async with _anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ) as client:
                message = await client.messages.create(
                    model=_MODEL_ID,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            text = message.content[0].text if message.content else ""
            text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
            text = re.sub(r"\n?```\s*$", "", text.strip())
            return text.strip()

        except _AnthropicRateLimitError as exc:
            last_exc = exc
            if attempt < _MAX_API_RETRIES:
                wait = min(2 ** (attempt + 1), 30)  # 2s, 4s, 8s, … capped at 30s
                logger.warning(
                    "reviewer.rate_limit_backoff",
                    extra={"attempt": attempt, "wait_seconds": wait, "run_id": run_id},
                )
                await asyncio.sleep(wait)

        except _AnthropicAPIError as exc:
            last_exc = exc
            if attempt < _MAX_API_RETRIES:
                await asyncio.sleep(1)

    raise last_exc or RuntimeError("Anthropic API failed after retries")


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _build_pass_output(
    verdict_dict: dict[str, Any],
    inp: ReviewerInput,
    *,
    degraded: bool = False,
) -> ReviewerOutput:
    raw_conf = verdict_dict.get("confidence", "MEDIUM")
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = (
        raw_conf if raw_conf in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
    )
    out = ReviewerOutput(
        run_id=inp.run_id,
        success=True,
        approved=True,
        verdict="pass",
        confidence=confidence,
        endorsement_note=verdict_dict.get("endorsement_note"),
    )
    if degraded:
        out.warnings.append("citation_check_skipped_no_sources")
    return out


def _build_fail_output(
    verdict_dict: dict[str, Any],
    inp: ReviewerInput,
) -> ReviewerOutput:
    raw_issues = verdict_dict.get("issues", [])
    issues = [
        ReviewIssue(
            type=i.get("type", "thin_content"),
            description=i.get("description", ""),
            suggestion=i.get("suggestion", ""),
        )
        for i in raw_issues
        if i.get("type") in ("citation_mismatch", "constraint_violated", "thin_content", "phi_detected")
    ]
    feedback = verdict_dict.get("feedback_for_summarizer", "")
    return ReviewerOutput(
        run_id=inp.run_id,
        success=True,  # agent succeeded; the FAIL verdict routes back to summarizer
        approved=False,
        verdict="fail",
        issues=issues,
        feedback=feedback,
    )


# ── Pre-flight data quality checks ────────────────────────────────────────────

def _auto_fail(
    inp: ReviewerInput,
    issue_type: Literal["citation_mismatch", "constraint_violated", "thin_content", "phi_detected"],
    description: str,
    suggestion: str,
    feedback: str,
) -> ReviewerOutput:
    return ReviewerOutput(
        run_id=inp.run_id,
        success=True,
        approved=False,
        verdict="fail",
        issues=[ReviewIssue(type=issue_type, description=description, suggestion=suggestion)],
        feedback=feedback,
    )


# ── Core review logic ──────────────────────────────────────────────────────────

async def _core_review(inp: ReviewerInput, *, tracer: Any) -> ReviewerOutput:
    retry_count = inp.retry_count or 0
    topic_id = inp.article.url or inp.run_id
    degraded_no_sources = not inp.sources

    with tracer.start_as_current_span("reviewer.review") as parent_span:
        parent_span.set_attribute("topic_id", topic_id)
        parent_span.set_attribute("run_id", inp.run_id)
        parent_span.set_attribute("retry_count", retry_count)
        parent_span.set_attribute("model_id", _MODEL_ID)
        # verdict and issue_count set after LLM call; never record content

        # ── PHI check (belt-and-suspenders) ───────────────────────────────────
        with tracer.start_as_current_span("reviewer.phi_check") as phi_span:
            combined = inp.summary + " " + " ".join(inp.key_points)
            phi_detections = detect_phi_in_output(combined)
            phi_detected = len(phi_detections) > 0
            phi_span.set_attribute("phi_detected", phi_detected)

        if phi_detected:
            logger.warning(
                "phi_detected_in_reviewer_input",
                extra={
                    "run_id": inp.run_id,
                    "phi_entity_count": len(phi_detections),
                    "agent_id": _AGENT_ID,
                    # Never log summary content
                },
            )
            result = _auto_fail(
                inp,
                issue_type="phi_detected",
                description="Summary contains personal health information",
                suggestion="Remove all personal health identifiers before resubmitting",
                feedback=(
                    "Summary contains personal health information that must be removed. "
                    "Do not include SSNs, NHS numbers, medical licence numbers, or other "
                    "PHI identifiers in the summary."
                ),
            )
            parent_span.set_attribute("verdict", "fail")
            parent_span.set_attribute("issue_count", 1)
            parent_span.add_event("reviewer.fail", attributes={"issue_types": ["phi_detected"]})
            return result

        # ── Data quality pre-checks ────────────────────────────────────────────
        if not inp.summary.strip():
            result = _auto_fail(
                inp,
                issue_type="thin_content",
                description="Summary is empty",
                suggestion="Produce a substantive summary of the source material",
                feedback="The summary was empty. Produce a substantive summary with at least 3 distinct key points.",
            )
            parent_span.set_attribute("verdict", "fail")
            parent_span.set_attribute("issue_count", 1)
            parent_span.add_event("reviewer.fail", attributes={"issue_types": ["thin_content"]})
            return result

        if not inp.citations and not degraded_no_sources:
            result = _auto_fail(
                inp,
                issue_type="citation_mismatch",
                description="Summary contains no citations",
                suggestion="Add inline [N] citation markers and a matching citations list",
                feedback=(
                    "The summary contained no citations. Every factual claim must be grounded "
                    "with an inline [N] marker and a matching citation entry referencing one "
                    "of the provided source URLs."
                ),
            )
            parent_span.set_attribute("verdict", "fail")
            parent_span.set_attribute("issue_count", 1)
            parent_span.add_event("reviewer.fail", attributes={"issue_types": ["citation_mismatch"]})
            return result

        # ── Build prompt ───────────────────────────────────────────────────────
        prompt_data = load_prompt("reviewer", _PROMPT_VERSION)
        system_prompt = get_system_prompt("reviewer", _PROMPT_VERSION)
        user_template: str = prompt_data.get("user_template", "")

        key_points_block = _format_key_points(inp.key_points)
        citations_block = _format_citations(inp.citations)
        sources_block = _format_sources(inp.sources)
        constraints_block = _format_constraints(inp.constraints)
        retry_context = _build_retry_context(retry_count)

        user_prompt = user_template.format(
            summary=inp.summary,
            key_points_block=key_points_block,
            citations_block=citations_block,
            citation_count=len(inp.citations),
            sources_block=sources_block,
            source_count=len(inp.sources),
            constraints_block=constraints_block,
            retry_context=retry_context,
        )
        token_estimate = (len(system_prompt) + len(user_prompt)) // 4

        # ── LLM call with JSON retry ───────────────────────────────────────────
        parsed: dict[str, Any] | None = None
        llm_error: PipelineError | None = None

        for json_attempt in range(_MAX_JSON_RETRIES + 1):
            with tracer.start_as_current_span("reviewer.llm_call") as llm_span:
                llm_span.set_attribute("model_id", _MODEL_ID)
                llm_span.set_attribute("prompt_version", _PROMPT_VERSION)
                llm_span.set_attribute("token_estimate", token_estimate)
                llm_span.set_attribute("retry_count", retry_count)

                try:
                    raw = await _call_claude(user_prompt, system_prompt, run_id=inp.run_id)
                except (_AnthropicRateLimitError, _AnthropicAPIError) as exc:
                    # Exhausted internal retries inside _call_claude — surface as RETRYABLE
                    llm_error = PipelineError(
                        error_code=INVALID_LLM_RESPONSE,
                        severity=ErrorSeverity.RETRYABLE,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message="Anthropic API error during review",
                        retry_hint="Retry; check ANTHROPIC_API_KEY and connectivity",
                        context={"error_type": type(exc).__name__, "retry_count": retry_count},
                    )
                    record_span_error(llm_span, llm_error)
                    break  # exit the JSON retry loop; surface RETRYABLE to orchestrator
                # Other exceptions (RuntimeError, ImportError, etc.) are NOT caught here.
                # They propagate up through _core_review to run_agent's outer handler,
                # which returns DEGRADED so the pipeline can continue unreviewed.

            if llm_error:
                break

            parsed = _parse_verdict(raw)
            if parsed is not None:
                break

            # JSON parse failed — retry unless exhausted
            if json_attempt < _MAX_JSON_RETRIES:
                logger.warning(
                    "reviewer_json_parse_failed_retrying",
                    extra={"run_id": inp.run_id, "json_attempt": json_attempt},
                )
            else:
                llm_error = PipelineError(
                    error_code=INVALID_LLM_RESPONSE,
                    severity=ErrorSeverity.RETRYABLE,
                    agent_id=_AGENT_ID,
                    run_id=inp.run_id,
                    message="Reviewer returned unparseable JSON after retries",
                    retry_hint="Retry; reinforce JSON-only output in system prompt",
                    context={"json_attempts": _MAX_JSON_RETRIES + 1, "retry_count": retry_count},
                )

        if llm_error:
            log_pipeline_error(logger, llm_error)
            record_span_error(parent_span, llm_error)
            return ReviewerOutput(run_id=inp.run_id, success=False, error=llm_error)

        # ── Map verdict to output ──────────────────────────────────────────────
        verdict_str = parsed.get("verdict", "").lower()  # type: ignore[union-attr]

        if verdict_str == "pass":
            result = _build_pass_output(parsed, inp, degraded=degraded_no_sources)  # type: ignore[arg-type]
        elif verdict_str == "fail":
            result = _build_fail_output(parsed, inp)  # type: ignore[arg-type]
        else:
            # Unrecognised verdict — treat as soft pass to avoid blocking the pipeline
            logger.warning(
                "reviewer_unrecognised_verdict",
                extra={"run_id": inp.run_id, "verdict_str": verdict_str},
            )
            result = ReviewerOutput(
                run_id=inp.run_id,
                success=True,
                approved=True,
                verdict="pass",
                confidence="LOW",
                endorsement_note="reviewer returned unrecognised verdict — defaulting to pass",
                warnings=["unrecognised_verdict"],
            )

        # ── OTel span finalisation ─────────────────────────────────────────────
        parent_span.set_attribute("verdict", result.verdict)
        parent_span.set_attribute("issue_count", len(result.issues))

        if result.verdict == "fail":
            issue_types = [i.type for i in result.issues]
            # Never record issue descriptions or feedback text in spans
            parent_span.add_event("reviewer.fail", attributes={"issue_types": issue_types})

        if degraded_no_sources:
            parent_span.set_attribute("degraded_reason", "no_sources")

    logger.info(
        "reviewer_complete",
        extra={
            "run_id": inp.run_id,
            "verdict": result.verdict,
            "issue_count": len(result.issues),
            "retry_count": retry_count,
            "degraded_no_sources": degraded_no_sources,
            # Never log feedback text, issue descriptions, or summary content
        },
    )
    return result


# ── LangSmith-traced wrappers ──────────────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="reviewer",
    tags=["agent:reviewer"],
)
async def _traced_run_initial(inp: ReviewerInput, *, tracer: Any) -> ReviewerOutput:
    """LangSmith run name: 'reviewer' — initial (non-retry) calls."""
    return await _core_review(inp, tracer=tracer)


@_ls_traceable(
    run_type="chain",
    name="reviewer-retry",
    tags=["agent:reviewer", "retry"],
)
async def _traced_run_retry(inp: ReviewerInput, *, tracer: Any) -> ReviewerOutput:
    """LangSmith run name: 'reviewer-retry' — summarizer-retry calls."""
    return await _core_review(inp, tracer=tracer)


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: ReviewerInput,
    *,
    tracer: Any = None,
) -> ReviewerOutput:
    """
    Public entry point called by main.py and the Phase 2 orchestrator.

    MODEL_OVERRIDE is not applied — the reviewer always uses claude-sonnet-4-6.
    If inp.retry_count > 0, the LangSmith run is named 'reviewer-retry'.

    On unrecoverable error, returns DEGRADED ReviewerOutput with approved=True
    so the pipeline can proceed unreviewed rather than stalling.
    """
    load_dotenv()
    setup_telemetry("news-mas-reviewer")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("reviewer")
    retry_count = inp.retry_count or 0

    make_run_config(
        "reviewer",
        _PROMPT_VERSION,
        run_id=inp.run_id,
        topic_id=inp.article.url or None,
    )

    logger.info(
        "reviewer_start",
        extra={
            "run_id": inp.run_id,
            "retry_count": retry_count,
            "source_count": len(inp.sources),
            "citation_count": len(inp.citations),
            "constraint_count": len(inp.constraints),
            "model_id": _MODEL_ID,
            "langsmith_active": langsmith_active,
            # Never log summary content, topic text, or constraints
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
            return ReviewerOutput(run_id=inp.run_id, success=False, error=auth_error)

    # ── Route to correctly named LangSmith trace, with DEGRADED fallback ──────
    try:
        if retry_count > 0:
            return await _traced_run_retry(inp, tracer=active_tracer)
        return await _traced_run_initial(inp, tracer=active_tracer)

    except Exception as exc:
        # Unexpected error in the review itself — degrade gracefully
        logger.error(
            "reviewer_unexpected_error",
            extra={
                "run_id": inp.run_id,
                "error_type": type(exc).__name__,
                # Never log exc message — may contain content
            },
        )
        degraded_error = PipelineError(
            error_code=REVIEWER_UNAVAILABLE,
            severity=ErrorSeverity.DEGRADED,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="Reviewer encountered an unexpected error; proceeding unreviewed",
            retry_hint="Check reviewer logs for error_type details",
            context={"error_type": type(exc).__name__, "retry_count": retry_count},
        )
        log_pipeline_error(logger, degraded_error)
        return ReviewerOutput(
            run_id=inp.run_id,
            success=True,
            approved=True,
            verdict="pass",
            confidence="LOW",
            endorsement_note="reviewer_unavailable — proceeding unreviewed",
            warnings=["reviewer_unavailable"],
            error=degraded_error,
        )
