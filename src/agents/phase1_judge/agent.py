"""
Phase 1 Judge — independent single-pass review of the selector's topic choices.

The judge sees everything the selector saw (all candidates, selector's selections
and rejections with reasoning) and makes one decision: ENDORSE or ADJUST.
Adjustments are targeted swaps within the existing candidate pool — the judge
does not re-rank from scratch.

Injection defence:
  - Topic text and reasoning treated as data, never as instructions
  - Hardened system preamble on every call
  - Topic text and article content NEVER appear in logs or OTel spans
  - Adjustments logged with topic IDs and reason_codes only

Single-pass design:
  - No retry loop with the selector — judge decides once
  - On LLM failure after retries: DEGRADED, return selector's original choices

OTel spans emitted:
  phase1_judge.evaluate  (parent) → run_id, candidate_count, selected_count,
                                    verdict, adjustment_count
  phase1_judge.llm_call  (child)  → model_id, prompt_version, input_token_estimate
  phase1_judge.adjustment (event) → swapped_out_topic_id, swapped_in_topic_id,
                                    reason_code (never reason text)

Security invariants:
  - Topic text NEVER in spans, logs, or error messages
  - Adjustments logged by topic_id + reason_code only, never reasoning text
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
except ImportError:
    _anthropic = None  # type: ignore[assignment]

from src.common.agent_bootstrap import resolve_model
from src.auth.token_service import validate_token
from src.common.data_quality import DataQualityValidator
from src.common.error_codes import (
    AUTH_FAILED,
    INVALID_LLM_RESPONSE,
    MODEL_NOT_FOUND,
    NO_VIABLE_CANDIDATES,
    OLLAMA_UNAVAILABLE,
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
    CandidateConfidence,
    JudgeAdjustment,
    JudgedArticle,
    JudgedTopic,
    Phase1JudgeInput,
    Phase1JudgeOutput,
    RejectedTopic,
    ScoredFilteredTopic,
    SelectedTopic,
)

_DEFAULT_MODEL = "gemma4:e4b"
_AGENT_ID = "phase1-judge"
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "judge.phase1"
_MAX_JSON_RETRIES = 2

_validator = DataQualityValidator(agent_id=_AGENT_ID)
logger = get_logger(__name__)


# ── AAP token validation ───────────────────────────────────────────────────────

def _validate_aap_token(token: str) -> dict[str, Any]:
    """Validate AAP token and assert judge.phase1 capability."""
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


# ── Ollama client ──────────────────────────────────────────────────────────────

def _get_ollama_client() -> Any:
    if _ollama is None:
        raise RuntimeError("ollama package is not installed")
    return _ollama.AsyncClient(
        host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


# ── Judge prompt payload builder ───────────────────────────────────────────────

def _build_judge_payload(
    all_candidates: list[ScoredFilteredTopic],
    selected: list[SelectedTopic],
    rejected: list[RejectedTopic],
    candidate_confidence: Optional[CandidateConfidence],
    run_id: str,
    since_date: Optional[str],
) -> str:
    """
    Build the formatted context block for the judge's user turn.

    Topic text is included for the LLM's category reasoning but must never
    appear in logs, spans, or error messages. This is the only callsite that
    forwards topic_text to the LLM.
    """
    selected_ids = {s.topic_id for s in selected}
    selected_by_id = {s.topic_id: s for s in selected}
    rejected_by_id = {r.topic_id: r for r in rejected}

    lines: list[str] = [
        "JUDGE CONTEXT",
        "=============",
        f"Run ID: {run_id}",
    ]
    if since_date:
        lines.append(f"Since date: {since_date}")

    if candidate_confidence:
        inj = candidate_confidence.injection_detected_count
        lines += [
            "",
            f"Batch source confidence: {candidate_confidence.overall_confidence}",
            f"  Full content: {candidate_confidence.full_content_count} / "
            f"{candidate_confidence.total_results} results",
            f"  Injections detected: {inj}" + (" ← RED FLAG" if inj > 0 else ""),
        ]

    lines += ["", f"ALL CANDIDATES ({len(all_candidates)} topics):"]
    for idx, topic in enumerate(all_candidates, start=1):
        status = "SELECTED" if topic.topic_id in selected_ids else "REJECTED"
        kept = len(topic.kept_results)
        removed = len(topic.removal_reasons)
        total = kept + removed
        lines += [
            f"",
            f"[{idx}] Topic ID  : {topic.topic_id}  [{status}]",
            f"    Query     : {topic.topic_text}",
            f"    Heat score: {topic.heat_score:.4f}  |  Confidence: {topic.confidence}",
            f"    Results   : {kept} kept / {total} total ({removed} removed by filter)",
        ]

    lines += ["", f"SELECTOR'S SELECTIONS ({len(selected)} topic(s)):"]
    for s in sorted(selected, key=lambda x: x.rank):
        lines += [
            f"",
            f"  Rank {s.rank}: {s.topic_id}",
            f"    Reasoning  : \"{s.selection_reasoning}\"",
            f"    Confidence : {s.confidence_assessment}",
        ]

    lines += ["", f"SELECTOR'S REJECTIONS ({len(rejected)} topic(s)):"]
    for r in rejected:
        lines += [f"  - {r.topic_id}: \"{r.reject_reason}\""]

    return "\n".join(lines)


# ── Build judged_articles for Phase 2 ─────────────────────────────────────────

def _build_judged_articles(
    final_selected: list[JudgedTopic],
    all_candidates: list[ScoredFilteredTopic],
) -> list[JudgedArticle]:
    """Flatten selected topics' kept_results into JudgedArticle list for Phase 2."""
    candidate_map = {t.topic_id: t for t in all_candidates}
    approved: list[JudgedArticle] = []
    selected_ids = {jt.topic_id for jt in final_selected}

    for jt in final_selected:
        topic = candidate_map.get(jt.topic_id)
        if topic is None:
            continue
        note = jt.judge_reasoning or "approved by phase1 judge"
        for sa in topic.kept_results:
            approved.append(JudgedArticle(
                article=sa.article,
                approved=True,
                reason=note,
            ))

    # Mark non-selected articles as rejected
    for topic in all_candidates:
        if topic.topic_id not in selected_ids:
            for sa in topic.kept_results:
                approved.append(JudgedArticle(
                    article=sa.article,
                    approved=False,
                    reason="topic not selected by phase1 judge",
                ))

    return approved


# ── LangSmith-traced LLM calls ────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="phase1-judge-gemma4",
    tags=["phase1-judge", "gemma4", "local"],
    metadata={"model": "gemma4:e4b"},
)
async def _call_ollama(
    prompt: str,
    system: str,
    run_id: str,
    *,
    model_id: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """LangSmith-traced Ollama inference call."""
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
    name="phase1-judge-anthropic",
    tags=["phase1-judge", "anthropic", "cloud"],
    metadata={"provider": "anthropic"},
)
async def _call_anthropic(
    prompt: str,
    system: str,
    run_id: str,
    *,
    model_id: str,
) -> dict[str, Any]:
    """LangSmith-traced Anthropic inference call. Returns Ollama-compatible shape."""
    if _anthropic is None:
        raise RuntimeError("anthropic package is not installed")
    async with _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) as client:
        message = await client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    text = message.content[0].text if message.content else ""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return {"message": {"content": text.strip()}}


# ── Fallback: return selector's original choices unchanged ─────────────────────

def _selector_fallback(
    inp: Phase1JudgeInput,
    error: PipelineError,
) -> Phase1JudgeOutput:
    """
    DEGRADED fallback when the judge cannot produce a verdict.
    Returns selector's original choices with source='selector'.
    """
    fallback = [
        JudgedTopic(topic_id=s.topic_id, rank=s.rank, source="selector")
        for s in sorted(inp.selected, key=lambda x: x.rank)
    ]
    judged = _build_judged_articles(fallback, inp.all_candidates)
    return Phase1JudgeOutput(
        run_id=inp.run_id,
        success=True,  # pipeline continues in DEGRADED mode
        error=error,
        verdict="endorsed",
        final_selected=fallback,
        adjustments=[],
        endorsement_note="Judge unavailable — selector choices returned unchanged.",
        overall_confidence="LOW",
        judged_articles=judged,
        warnings=["judge_unavailable:DEGRADED"],
    )


# ── LangSmith-traced judge run ────────────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="phase1_judge",
    tags=["agent:phase1_judge"],
)
async def _traced_run(
    inp: Phase1JudgeInput,
    *,
    model_id: str,
    provider: str,
    prompt_version: str,
    tracer: Any,
) -> Phase1JudgeOutput:
    """LangSmith-traced body. inp is captured as run input for tracing."""
    prompt_data = load_prompt("phase1_judge", prompt_version)
    system_prompt = get_system_prompt("phase1_judge", prompt_version)
    user_template: str = prompt_data.get("user_template", "")

    # ── Validate — need at least one candidate and one selection ──────────
    if not inp.all_candidates and not inp.selected and not inp.selected_articles:
        error = PipelineError(
            error_code=NO_VIABLE_CANDIDATES,
            severity=ErrorSeverity.FATAL,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="Phase 1 Judge received no candidates or selections",
            retry_hint=(
                "Ensure selector produced output before calling phase1_judge"
            ),
            context={
                "all_candidates_count": 0,
                "selected_count": 0,
            },
        )
        log_pipeline_error(logger, error)
        return Phase1JudgeOutput(
            run_id=inp.run_id,
            success=False,
            error=error,
            verdict="endorsed",
            final_selected=[],
            judged_articles=[],
        )

    # ── DEGRADED if all candidates are LOW confidence ─────────────────────
    warnings: list[str] = []
    all_low = bool(inp.all_candidates) and all(
        t.confidence == "LOW" for t in inp.all_candidates
    )
    if all_low:
        warnings.append("all_candidates_low_confidence:DEGRADED")
        logger.warning(
            "phase1_judge_all_low_confidence",
            extra={
                "run_id": inp.run_id,
                "candidate_count": len(inp.all_candidates),
            },
        )

    # ── Build prompt payload ───────────────────────────────────────────────
    candidates_payload = _build_judge_payload(
        inp.all_candidates,
        inp.selected,
        inp.rejected,
        inp.candidate_confidence,
        inp.run_id,
        inp.since_date,
    )
    user_prompt = user_template.format(candidates_payload=candidates_payload)
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4

    # ── Candidate index for validation ────────────────────────────────────
    known_ids = {t.topic_id for t in inp.all_candidates}
    expected_count = len(inp.selected)

    ollama_resp: dict[str, Any] | None = None
    last_error: PipelineError | None = None
    parsed: dict[str, Any] | None = None

    with tracer.start_as_current_span("phase1_judge.evaluate") as eval_span:
        eval_span.set_attribute("run_id", inp.run_id)
        eval_span.set_attribute("candidate_count", len(inp.all_candidates))
        eval_span.set_attribute("selected_count", len(inp.selected))
        # Never record topic text in spans

        for attempt in range(1 + _MAX_JSON_RETRIES):
            with tracer.start_as_current_span("phase1_judge.llm_call") as llm_span:
                llm_span.set_attribute("model_id", model_id)
                llm_span.set_attribute("prompt_version", prompt_version)
                llm_span.set_attribute("input_token_estimate", token_estimate)
                llm_span.set_attribute("attempt", attempt)

                try:
                    if provider == "anthropic":
                        ollama_resp = await _call_anthropic(
                            user_prompt, system_prompt, inp.run_id, model_id=model_id,
                        )
                    else:
                        ollama_resp = await _call_ollama(
                            user_prompt, system_prompt, inp.run_id, model_id=model_id,
                        )
                except _OllamaResponseError as exc:
                    status = getattr(exc, "status_code", -1)
                    msg_lower = str(exc).lower()
                    is_missing = (
                        status == 404
                        or ("not found" in msg_lower and "model" in msg_lower)
                        or ("pull" in msg_lower and "model" in msg_lower)
                    )
                    code = MODEL_NOT_FOUND if is_missing else OLLAMA_UNAVAILABLE
                    last_error = PipelineError(
                        error_code=code,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message=(
                            f"Ollama {'model not available' if is_missing else 'API error'}: "
                            f"{model_id}"
                        ),
                        retry_hint=(
                            f"Run 'ollama pull {model_id}'" if is_missing
                            else "Check Ollama server status at OLLAMA_BASE_URL"
                        ),
                        context={"model_id": model_id, "status_code": status},
                    )
                    record_span_error(llm_span, last_error)
                    record_span_error(eval_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return Phase1JudgeOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        verdict="endorsed",
                        final_selected=[],
                        judged_articles=[],
                    )

                except (ConnectionError, OSError, TimeoutError) as exc:
                    last_error = PipelineError(
                        error_code=OLLAMA_UNAVAILABLE,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message="Ollama service is unreachable",
                        retry_hint="Ensure Ollama is running at OLLAMA_BASE_URL",
                        context={"error_type": type(exc).__name__},
                    )
                    record_span_error(llm_span, last_error)
                    record_span_error(eval_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return Phase1JudgeOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        verdict="endorsed",
                        final_selected=[],
                        judged_articles=[],
                    )

                except Exception as exc:
                    last_error = PipelineError(
                        error_code=OLLAMA_UNAVAILABLE,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message="Unexpected error calling LLM",
                        retry_hint="Check LLM backend connectivity and model availability",
                        context={"error_type": type(exc).__name__, "provider": provider},
                    )
                    record_span_error(llm_span, last_error)
                    record_span_error(eval_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return Phase1JudgeOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        verdict="endorsed",
                        final_selected=[],
                        judged_articles=[],
                    )

            # ── Parse JSON ─────────────────────────────────────────────────
            raw_content = (ollama_resp or {}).get("message", {}).get("content", "")
            try:
                candidate = json.loads(raw_content)
                if "verdict" not in candidate or "final_selected" not in candidate:
                    raise ValueError("Missing required keys: verdict, final_selected")
                parsed = candidate
                break  # successful parse
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = PipelineError(
                    error_code=INVALID_LLM_RESPONSE,
                    severity=ErrorSeverity.RETRYABLE,
                    agent_id=_AGENT_ID,
                    run_id=inp.run_id,
                    message="LLM returned invalid or unparseable JSON",
                    retry_hint="Retry; Ollama format=json should produce valid JSON",
                    context={
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )
                log_pipeline_error(logger, last_error)
                # Loop will retry if attempts remain

        # ── All retries failed — DEGRADED fallback ─────────────────────────
        if parsed is None:
            if last_error is None:
                last_error = PipelineError(
                    error_code=INVALID_LLM_RESPONSE,
                    severity=ErrorSeverity.RETRYABLE,
                    agent_id=_AGENT_ID,
                    run_id=inp.run_id,
                    message="Judge response could not be parsed after retries",
                    retry_hint="Retry the phase1_judge call",
                    context={"candidate_count": len(inp.all_candidates)},
                )
            log_pipeline_error(logger, last_error)
            eval_span.set_attribute("verdict", "fallback_degraded")
            eval_span.set_attribute("adjustment_count", 0)
            return _selector_fallback(inp, last_error)

        # ── Validate and apply judge decisions ─────────────────────────────
        raw_verdict: str = parsed.get("verdict", "endorsed")
        verdict: Literal["endorsed", "adjusted"] = (
            "adjusted" if raw_verdict == "adjusted" else "endorsed"
        )
        raw_final: list[dict] = parsed.get("final_selected", [])
        raw_adjustments: list[dict] = parsed.get("adjustments", [])
        endorsement_note: Optional[str] = parsed.get("endorsement_note") or None
        raw_overall: str = parsed.get("overall_confidence", "LOW")
        overall_conf: Literal["HIGH", "MEDIUM", "LOW"] = (
            raw_overall if raw_overall in ("HIGH", "MEDIUM", "LOW") else "LOW"
        )

        # Build final_selected — only accept topic_ids present in all_candidates
        final_selected: list[JudgedTopic] = []
        seen_ids: set[str] = set()
        for entry in raw_final:
            tid = entry.get("topic_id", "")
            if not tid or tid not in known_ids or tid in seen_ids:
                continue
            src = entry.get("source", "selector")
            source_val: Literal["selector", "judge"] = (
                "judge" if src == "judge" else "selector"
            )
            final_selected.append(JudgedTopic(
                topic_id=tid,
                rank=int(entry.get("rank", len(final_selected) + 1)),
                source=source_val,
                judge_reasoning=(
                    str(entry["judge_reasoning"])
                    if entry.get("judge_reasoning") else None
                ),
            ))
            seen_ids.add(tid)

        # If judge returned nothing valid, fall back to selector choices
        if not final_selected:
            dq_error = PipelineError(
                error_code=INVALID_LLM_RESPONSE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                message="Judge final_selected contained no valid topic_ids",
                retry_hint="Retry; all returned topic_ids must match input candidates",
                context={"known_ids_count": len(known_ids)},
            )
            log_pipeline_error(logger, dq_error)
            eval_span.set_attribute("verdict", "fallback_degraded")
            eval_span.set_attribute("adjustment_count", 0)
            return _selector_fallback(inp, dq_error)

        # Build adjustments — only valid swaps within known_ids
        valid_reason_codes = {
            "confidence_upgrade", "diversity_improvement", "quality_concern"
        }
        adjustments: list[JudgeAdjustment] = []
        for adj in raw_adjustments:
            removed = adj.get("removed_topic_id", "")
            added = adj.get("added_topic_id", "")
            reason = adj.get("reason_code", "")
            if (
                removed in known_ids
                and added in known_ids
                and removed != added
                and reason in valid_reason_codes
            ):
                adjustments.append(JudgeAdjustment(
                    action="swap",
                    removed_topic_id=removed,
                    added_topic_id=added,
                    reason_code=reason,  # type: ignore[arg-type]
                ))

        # If verdict is "adjusted" but no valid adjustments → treat as endorsed
        if verdict == "adjusted" and not adjustments:
            verdict = "endorsed"

        # Emit OTel events for each adjustment (IDs + reason_code only, never text)
        for adj in adjustments:
            eval_span.add_event(
                "phase1_judge.adjustment",
                attributes={
                    "swapped_out_topic_id": adj.removed_topic_id,
                    "swapped_in_topic_id": adj.added_topic_id,
                    "reason_code": adj.reason_code,
                },
            )

        eval_span.set_attribute("verdict", verdict)
        eval_span.set_attribute("adjustment_count", len(adjustments))

    # ── Build judged_articles flat list for Phase 2 ───────────────────────
    judged_articles = _build_judged_articles(final_selected, inp.all_candidates)

    logger.info(
        "phase1_judge_complete",
        extra={
            "run_id": inp.run_id,
            "candidate_count": len(inp.all_candidates),
            "selected_count": len(inp.selected),
            "final_selected_count": len(final_selected),
            "verdict": verdict,
            "adjustment_count": len(adjustments),
            "overall_confidence": overall_conf,
            "model_id": model_id,
            # Never log topic text or reasoning
        },
    )

    return Phase1JudgeOutput(
        run_id=inp.run_id,
        success=True,
        verdict=verdict,
        final_selected=final_selected,
        adjustments=adjustments,
        endorsement_note=endorsement_note,
        overall_confidence=overall_conf,
        judged_articles=judged_articles,
        warnings=warnings,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: Phase1JudgeInput,
    *,
    tracer: Any = None,
    model_id: str | None = None,
    model_provider: str | None = None,
) -> Phase1JudgeOutput:
    """
    Public entry point. ``model_id`` and ``model_provider`` are optional —
    main.py injects them from AgentConfig. MODEL_OVERRIDE is applied via
    resolve_model(); summarizer and reviewer are exempt.
    """
    load_dotenv()
    setup_telemetry("news-mas-phase1-judge")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("phase1-judge")

    prompt_version = _PROMPT_VERSION
    provider, effective_model = resolve_model(
        model_id or _DEFAULT_MODEL,
        model_provider or "ollama",
        _AGENT_ID,
    )

    make_run_config("phase1_judge", prompt_version, run_id=inp.run_id)
    logger.info(
        "phase1_judge_start",
        extra={
            "run_id": inp.run_id,
            "all_candidates_count": len(inp.all_candidates),
            "selected_count": len(inp.selected),
            "model_id": effective_model,
            "langsmith_active": langsmith_active,
            # Never log topic text
        },
    )

    if inp.aap_token:
        try:
            _validate_aap_token(inp.aap_token)
        except PermissionError as exc:
            error = PipelineError(
                error_code=AUTH_FAILED,
                severity=ErrorSeverity.FATAL,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                message=str(exc),
                context={},
            )
            log_pipeline_error(logger, error)
            return Phase1JudgeOutput(
                run_id=inp.run_id,
                success=False,
                error=error,
                verdict="endorsed",
                final_selected=[],
                judged_articles=[],
            )

    return await _traced_run(
        inp,
        model_id=effective_model,
        provider=provider,
        prompt_version=prompt_version,
        tracer=active_tracer,
    )
