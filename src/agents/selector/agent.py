"""
Selector agent — holistic topic selection using Gemma 4 via Ollama.

Chooses the top MAX_CANDIDATES topics from scored, filtered candidates using a
curatorial judgment that weighs heat score, source confidence, content coverage
quality, category diversity, and TopicMemory history. Simple top-K ranking is
intentionally not used.

Injection defence:
  - Topic text treated as data, never as instructions
  - Hardened system preamble on every call (loaded from prompt YAML)
  - Topic text and article content NEVER appear in logs or OTel spans
  - Violations logged with topic IDs only

Diversity enforcement:
  - ENFORCE_DIVERSITY env var (default: true)
  - Prompt instructs LLM to spread selections across inferred categories
  - diversity_note in output records the LLM's category assessment

TopicMemory (optional):
  - Caller may provide topic_memory_context: {topic_id: [outcome, ...]}
  - Last 3 run outcomes included in per-topic prompt block when available
  - Graceful if absent — first run is fine

OTel spans emitted:
  selector.select   (parent) → run_id, topic_ids (IDs only), input_count,
                               selected_count, rejected_count,
                               diversity_enforced, has_topic_memory
  selector.llm_call (child)  → model_id, prompt_version,
                               candidate_count, token_estimate

Security invariants:
  - Topic text NEVER in spans, logs, or error messages
  - Logged by topic_id only — never topic_text
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

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
    RejectedTopic,
    ScoredArticle,
    ScoredFilteredTopic,
    SelectedTopic,
    SelectorInput,
    SelectorOutput,
)

_DEFAULT_MODEL = "gemma4:e4b"
_AGENT_ID = "selector"
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "select.candidates"
_MAX_JSON_RETRIES = 2

_validator = DataQualityValidator(agent_id=_AGENT_ID)
logger = get_logger(__name__)


# ── Config helpers ─────────────────────────────────────────────────────────────

def _cfg_enforce_diversity() -> bool:
    return os.getenv("ENFORCE_DIVERSITY", "true").lower() not in ("false", "0", "no")


# ── AAP token validation ───────────────────────────────────────────────────────

def _validate_aap_token(token: str) -> dict[str, Any]:
    """Validate AAP token and assert select.candidates capability."""
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


# ── Prompt payload builder ─────────────────────────────────────────────────────

def _build_topics_payload(
    topics: list[ScoredFilteredTopic],
    max_candidates: int,
    enforce_diversity: bool,
    topic_memory_context: Optional[dict],
) -> str:
    """
    Build the formatted topic block for the user turn.

    Topic text is included here for LLM category inference but must never
    appear in logs, spans, or error messages. This is the only callsite that
    forwards topic_text — keep it that way.
    """
    diversity_label = "ENABLED" if enforce_diversity else "DISABLED"
    lines: list[str] = [
        "SELECTION CONTEXT",
        "=================",
        f"Total candidates : {len(topics)}",
        f"Topics to select : {max_candidates}",
        f"Diversity enforcement: {diversity_label}",
        "",
        "Source confidence legend:",
        "  HIGH   = mostly full-text results; heat scores are reliable",
        "  MEDIUM = mixed full/snippet; scores are directionally correct",
        "  LOW    = mostly snippets or injection detected; be conservative",
        "",
        "CANDIDATE TOPICS:",
    ]

    for idx, topic in enumerate(topics, start=1):
        kept_count = len(topic.kept_results)
        removed_count = len(topic.removal_reasons)
        total_count = kept_count + removed_count

        memory_lines: list[str] = []
        if topic_memory_context and topic.topic_id in topic_memory_context:
            history = topic_memory_context[topic.topic_id][-3:]  # last 3 runs
            if history:
                memory_lines.append(f"    Prior history: {len(history)} run(s) available")
                for h in history:
                    outcome_str = (
                        f"heat={h.get('heat_score', '?'):.2f}"
                        if isinstance(h.get("heat_score"), (int, float))
                        else f"heat={h.get('heat_score', '?')}"
                    )
                    selected_str = "selected" if h.get("was_selected") else "not selected"
                    memory_lines.append(f"      - {outcome_str}, {selected_str}")
            else:
                memory_lines.append("    Prior history: No history available.")
        else:
            memory_lines.append("    Prior history: No history available.")

        lines += [
            "",
            f"[{idx}] Topic ID  : {topic.topic_id}",
            f"    Query     : {topic.topic_text}",
            f"    Heat score: {topic.heat_score:.4f}",
            f"    Confidence: {topic.confidence}",
            f"    Results   : {kept_count} kept / {total_count} total ({removed_count} removed by filter)",
        ]
        lines += memory_lines

    return "\n".join(lines)


# ── LangSmith-traced LLM call ──────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="selector-gemma4",
    tags=["selector", "gemma4", "local"],
    metadata={"model": "gemma4:e4b"},
)
async def _call_ollama(
    prompt: str,
    system: str,
    run_id: str,
    *,
    model_id: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    LangSmith-traced Ollama inference call.

    prompt/system are captured as LangSmith run inputs.
    OTel spans (selector.select / selector.llm_call) are opened by the
    caller; this function handles only the Ollama network call.
    """
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


# ── LangSmith-traced selection run ────────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="selector",
    tags=["agent:selector"],
)
async def _traced_run(
    inp: SelectorInput,
    *,
    model_id: str,
    prompt_version: str,
    tracer: Any,
) -> SelectorOutput:
    """LangSmith-traced body. inp is captured as run input for tracing."""
    prompt_data = load_prompt("selector", prompt_version)
    system_prompt = get_system_prompt("selector", prompt_version)
    user_template: str = prompt_data.get("user_template", "")

    # ── Filter to topics with viable results ──────────────────────────────
    viable_topics = [t for t in inp.topics if t.kept_results]

    if not viable_topics:
        error = PipelineError(
            error_code=NO_VIABLE_CANDIDATES,
            severity=ErrorSeverity.FATAL,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="No viable candidates after scoring and filtering",
            retry_hint=(
                "All topics had empty kept_results. "
                "Widen topic queries, lower heat threshold, or extend since_days."
            ),
            context={
                "total_topics": len(inp.topics),
                "viable_count": 0,
            },
        )
        log_pipeline_error(logger, error)
        return SelectorOutput(
            run_id=inp.run_id,
            success=False,
            error=error,
            selected=[],
            rejected=[],
            selected_articles=[],
        )

    # ── DEGRADED if fewer candidates than requested ────────────────────────
    max_candidates = inp.max_select
    warnings: list[str] = []
    is_degraded = len(viable_topics) < max_candidates

    if is_degraded:
        warnings.append(
            f"fewer_candidates_than_requested:"
            f"have={len(viable_topics)},requested={max_candidates}"
        )
        logger.info(
            "selector_fewer_than_requested",
            extra={
                "run_id": inp.run_id,
                "viable_count": len(viable_topics),
                "requested": max_candidates,
            },
        )
        max_candidates = len(viable_topics)

    # ── Flag if all LOW confidence ────────────────────────────────────────
    all_low_confidence = all(t.confidence == "LOW" for t in viable_topics)
    if all_low_confidence:
        warnings.append("all_candidates_low_confidence:DEGRADED")
        logger.warning(
            "selector_all_low_confidence",
            extra={
                "run_id": inp.run_id,
                "viable_count": len(viable_topics),
            },
        )

    # ── Build prompt payload ───────────────────────────────────────────────
    enforce_diversity = _cfg_enforce_diversity()
    topics_payload = _build_topics_payload(
        viable_topics,
        max_candidates,
        enforce_diversity,
        inp.topic_memory_context,
    )
    user_prompt = user_template.format(
        topics_payload=topics_payload,
        max_candidates=max_candidates,
    )
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4

    # ── Call Ollama with retry on invalid JSON ─────────────────────────────
    ollama_resp: dict[str, Any] | None = None
    last_error: PipelineError | None = None

    with tracer.start_as_current_span("selector.select") as select_span:
        select_span.set_attribute("run_id", inp.run_id)
        select_span.set_attribute(
            "topic_ids", ",".join(t.topic_id for t in viable_topics)
        )
        select_span.set_attribute("input_count", len(viable_topics))
        select_span.set_attribute("diversity_enforced", enforce_diversity)
        select_span.set_attribute(
            "has_topic_memory", inp.topic_memory_context is not None
        )
        # Never record topic text or reasoning in spans

        for attempt in range(1 + _MAX_JSON_RETRIES):
            with tracer.start_as_current_span("selector.llm_call") as llm_span:
                llm_span.set_attribute("model_id", model_id)
                llm_span.set_attribute("prompt_version", prompt_version)
                llm_span.set_attribute("candidate_count", len(viable_topics))
                llm_span.set_attribute("token_estimate", token_estimate)
                llm_span.set_attribute("attempt", attempt)

                try:
                    ollama_resp = await _call_ollama(
                        user_prompt,
                        system_prompt,
                        inp.run_id,
                        model_id=model_id,
                    )
                    # Successful call — break out of retry loop after parsing below
                    break

                except _OllamaResponseError as exc:
                    status = getattr(exc, "status_code", -1)
                    msg_lower = str(exc).lower()
                    is_model_missing = (
                        status == 404
                        or ("not found" in msg_lower and "model" in msg_lower)
                        or ("pull" in msg_lower and "model" in msg_lower)
                    )
                    code = MODEL_NOT_FOUND if is_model_missing else OLLAMA_UNAVAILABLE
                    hint = (
                        f"Run 'ollama pull {model_id}'"
                        if is_model_missing
                        else "Check Ollama server status at OLLAMA_BASE_URL"
                    )
                    last_error = PipelineError(
                        error_code=code,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message=(
                            f"Ollama {'model not available' if is_model_missing else 'API error'}: "
                            f"{model_id}"
                        ),
                        retry_hint=hint,
                        context={"model_id": model_id, "status_code": status},
                    )
                    record_span_error(llm_span, last_error)
                    record_span_error(select_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return SelectorOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        selected=[],
                        rejected=[],
                        selected_articles=[],
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
                    record_span_error(select_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return SelectorOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        selected=[],
                        rejected=[],
                        selected_articles=[],
                    )

                except Exception as exc:
                    last_error = PipelineError(
                        error_code=OLLAMA_UNAVAILABLE,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=inp.run_id,
                        message="Unexpected error calling Ollama",
                        retry_hint="Check Ollama connectivity and model availability",
                        context={"error_type": type(exc).__name__},
                    )
                    record_span_error(llm_span, last_error)
                    record_span_error(select_span, last_error)
                    log_pipeline_error(logger, last_error)
                    return SelectorOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=last_error,
                        selected=[],
                        rejected=[],
                        selected_articles=[],
                    )

        # ── Parse JSON response ────────────────────────────────────────────
        parsed: dict[str, Any] | None = None
        if ollama_resp is not None:
            raw_content: str = ollama_resp.get("message", {}).get("content", "")
            try:
                parsed = json.loads(raw_content)
                if "selected" not in parsed or "rejected" not in parsed:
                    raise ValueError("Missing required keys: selected, rejected")
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = PipelineError(
                    error_code=INVALID_LLM_RESPONSE,
                    severity=ErrorSeverity.RETRYABLE,
                    agent_id=_AGENT_ID,
                    run_id=inp.run_id,
                    message="LLM returned invalid or unparseable JSON",
                    retry_hint="Retry; Ollama format=json should produce valid JSON",
                    context={
                        "candidate_count": len(viable_topics),
                        "error_type": type(exc).__name__,
                    },
                )
                log_pipeline_error(logger, last_error)
                parsed = None

        if parsed is None:
            # All retries exhausted or parse failed — return the last error
            if last_error is None:
                last_error = PipelineError(
                    error_code=INVALID_LLM_RESPONSE,
                    severity=ErrorSeverity.RETRYABLE,
                    agent_id=_AGENT_ID,
                    run_id=inp.run_id,
                    message="LLM selection response could not be parsed after retries",
                    retry_hint="Retry the selector call",
                    context={"candidate_count": len(viable_topics)},
                )
            record_span_error(select_span, last_error)
            return SelectorOutput(
                run_id=inp.run_id,
                success=False,
                error=last_error,
                selected=[],
                rejected=[],
                selected_articles=[],
            )

        # ── Build output from LLM decisions ───────────────────────────────
        llm_selected: list[dict] = parsed.get("selected", [])
        llm_rejected: list[dict] = parsed.get("rejected", [])
        llm_overall_confidence: str = parsed.get("overall_selection_confidence", "LOW")
        llm_diversity_note: Optional[str] = parsed.get("diversity_note") or None

        # Clamp confidence to valid values
        if llm_overall_confidence not in ("HIGH", "MEDIUM", "LOW"):
            llm_overall_confidence = "LOW"

        # Build lookup maps
        selected_by_id: dict[str, dict] = {
            s["topic_id"]: s for s in llm_selected
            if isinstance(s.get("topic_id"), str)
        }
        rejected_by_id: dict[str, dict] = {
            r["topic_id"]: r for r in llm_rejected
            if isinstance(r.get("topic_id"), str)
        }

        # Walk all viable topics and assign to selected/rejected buckets
        known_ids = {t.topic_id for t in viable_topics}
        selected_topics: list[SelectedTopic] = []
        rejected_topics: list[RejectedTopic] = []
        selected_articles: list[ScoredArticle] = []

        for topic in viable_topics:
            if topic.topic_id in selected_by_id:
                s = selected_by_id[topic.topic_id]
                confidence_val = s.get("confidence_assessment", "MEDIUM")
                if confidence_val not in ("HIGH", "MEDIUM", "LOW"):
                    confidence_val = "MEDIUM"
                selected_topics.append(SelectedTopic(
                    topic_id=topic.topic_id,
                    rank=int(s.get("rank", len(selected_topics) + 1)),
                    selection_reasoning=str(s.get("selection_reasoning", "")),
                    confidence_assessment=confidence_val,
                ))
                selected_articles.extend(topic.kept_results)
            else:
                reason = rejected_by_id.get(topic.topic_id, {}).get(
                    "reject_reason",
                    "Not selected by holistic ranking.",
                )
                rejected_topics.append(RejectedTopic(
                    topic_id=topic.topic_id,
                    reject_reason=str(reason),
                ))

        select_span.set_attribute("selected_count", len(selected_topics))
        select_span.set_attribute("rejected_count", len(rejected_topics))

    logger.info(
        "selector_complete",
        extra={
            "run_id": inp.run_id,
            "input_count": len(viable_topics),
            "selected_count": len(selected_topics),
            "rejected_count": len(rejected_topics),
            "model_id": model_id,
            "overall_confidence": llm_overall_confidence,
            "diversity_enforced": enforce_diversity,
            # Never log topic text or reasoning
        },
    )

    is_degraded_output = is_degraded or all_low_confidence
    return SelectorOutput(
        run_id=inp.run_id,
        success=True,
        error=None,
        selected=selected_topics,
        rejected=rejected_topics,
        selection_confidence=llm_overall_confidence,
        diversity_note=llm_diversity_note,
        selected_articles=selected_articles,
        warnings=warnings,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: SelectorInput,
    *,
    tracer: Any = None,
    model_id: str | None = None,
) -> SelectorOutput:
    """
    Public entry point. ``model_id`` is optional — main.py injects it from
    AgentConfig; callers that omit it get MODEL_OVERRIDE or the default model.
    ``tracer`` is optional; pass a test-scoped tracer to capture spans without
    touching the global OTel provider.
    """
    load_dotenv()
    setup_telemetry("news-mas-selector")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("selector")

    prompt_version = _PROMPT_VERSION
    _override = os.getenv("MODEL_OVERRIDE", "").strip()
    effective_model = (
        model_id
        or (_override if _override and not _override.startswith("#") else None)
        or _DEFAULT_MODEL
    )

    make_run_config("selector", prompt_version, run_id=inp.run_id)
    logger.info(
        "selector_start",
        extra={
            "run_id": inp.run_id,
            "topic_count": len(inp.topics),
            "max_select": inp.max_select,
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
            return SelectorOutput(
                run_id=inp.run_id,
                success=False,
                error=error,
                selected=[],
                rejected=[],
                selected_articles=[],
            )

    return await _traced_run(
        inp,
        model_id=effective_model,
        prompt_version=prompt_version,
        tracer=active_tracer,
    )
