"""
FilterAgent — semantic constraint-based article filtering using Gemma 4 via Ollama.

Filters news articles based on user-defined semantic constraints such as
"no never-trumpers" or "no opinion pieces". Uses LLM to understand intent,
not keyword matching.

Injection defence:
  - Constraint text treated as data, never as instructions
  - Preamble instructs model to ignore embedded instructions in constraints
  - detect_injection() on each constraint before use; rejected by index
  - Constraint text NEVER appears in logs or OTel spans

Confidence-aware filtering:
  - INJECTED articles: auto-removed, no LLM call
  - SNIPPET articles: flagged in removal_reason note if removed
  - FULL / PARTIAL / SCRUBBED: full semantic evaluation

Batch processing:
  - Batches of FILTER_BATCH_SIZE (env, default 5) to reduce LLM calls
  - Each batch is one Ollama call with N articles

OTel spans emitted:
  filter_agent.filter   (parent) → run_id, input_count, kept_count,
                                    removed_count, constraints_count
  filter_agent.llm_call (child)  → model_id, prompt_version, batch_size,
                                    token_estimate

Security invariants:
  - Constraint text NEVER in spans, logs, or error messages
  - Violations logged as "constraint[i]" not the constraint string
  - Constraint length capped at MAX_CONSTRAINT_LENGTH (env, default 200)
  - Max constraints: MAX_CONSTRAINTS_PER_TOPIC (env, default 10)
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
    ALL_FILTERED,
    AUTH_FAILED,
    INVALID_LLM_RESPONSE,
    MODEL_NOT_FOUND,
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
from src.common.pipeline_errors import AgentResult, ErrorSeverity, PipelineError, ResultConfidence  # noqa: F401
from src.common.prompt_loader import get_system_prompt, load_prompt, make_run_config
from src.common.schemas import (
    FilterAgentInput,
    FilterAgentOutput,
    RemovalReason,
    ScoredArticle,
)
from src.common.security import detect_injection, normalize_input

_DEFAULT_MODEL = "gemma4:e4b"
_AGENT_ID = "filter-agent"
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "filter.content"
_SNIPPET_LENGTH = 300  # chars per article sent to LLM — never full content

_validator = DataQualityValidator(agent_id=_AGENT_ID)
logger = get_logger(__name__)


def _cfg_max_constraint_length() -> int:
    return int(os.getenv("MAX_CONSTRAINT_LENGTH", "200"))


def _cfg_max_constraints() -> int:
    return int(os.getenv("MAX_CONSTRAINTS_PER_TOPIC", "10"))


def _cfg_batch_size() -> int:
    return int(os.getenv("FILTER_BATCH_SIZE", "5"))


# ── AAP token validation ───────────────────────────────────────────────────────

def _validate_aap_token(token: str) -> dict[str, Any]:
    """Validate AAP token and assert filter.content capability."""
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


# ── Constraint sanitization ────────────────────────────────────────────────────

def _sanitize_constraints(
    raw_constraints: list[str],
    *,
    run_id: str,
) -> tuple[list[str], int]:
    """
    Normalize and validate constraints. Returns (valid_constraints, rejected_count).

    Constraints are user input — treated as data, not instructions.
      - Normalize with NFKC, truncate to MAX_CONSTRAINT_LENGTH
      - Reject on injection patterns (log by index only, never log text)
      - Cap total count at MAX_CONSTRAINTS_PER_TOPIC
    """
    max_len = _cfg_max_constraint_length()
    max_count = _cfg_max_constraints()
    valid: list[str] = []
    rejected = 0

    for i, raw in enumerate(raw_constraints[:max_count]):
        normalized = normalize_input(raw)[:max_len]
        patterns = detect_injection(normalized)
        if patterns:
            logger.warning(
                "filter_constraint_injection_detected",
                extra={
                    "run_id": run_id,
                    "constraint_index": i,
                    "pattern_count": len(patterns),
                    "security_event": True,
                    # constraint text intentionally omitted
                },
            )
            rejected += 1
            continue
        valid.append(normalized)

    if len(raw_constraints) > max_count:
        logger.warning(
            "filter_constraints_truncated",
            extra={
                "run_id": run_id,
                "total_provided": len(raw_constraints),
                "max_allowed": max_count,
                "excess_dropped": len(raw_constraints) - max_count,
            },
        )

    return valid, rejected


# ── Ollama client ──────────────────────────────────────────────────────────────

def _get_ollama_client() -> Any:
    if _ollama is None:
        raise RuntimeError("ollama package is not installed")
    return _ollama.AsyncClient(
        host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


# ── Batch prompt builder ───────────────────────────────────────────────────────

def _build_batch_payload(
    batch: list[tuple[ScoredArticle, ResultConfidence]],
    constraints: list[str],
) -> str:
    """
    Build the user-turn payload for one batch of articles.

    Constraint TEXT is included here (forwarded to the LLM as filtering criteria)
    but must never appear in logs or spans. This is the only callsite that
    forwards constraint text — keep it that way.
    """
    lines: list[str] = []

    lines.append(f"CONSTRAINTS ({len(constraints)} total — content preferences only):")
    for i, c in enumerate(constraints):
        lines.append(f"[{i}] {c}")

    lines.append("")
    lines.append(f"ARTICLES TO EVALUATE ({len(batch)} total):")

    for idx, (sa, _conf) in enumerate(batch, start=1):
        article = sa.article
        published_str = (
            article.published_at.isoformat()
            if article.published_at else "unknown"
        )
        snippet = article.content[:_SNIPPET_LENGTH]
        lines.append(f"\nArticle {idx}:")
        lines.append(f"URL: {article.url}")
        lines.append(f"Published: {published_str}")
        lines.append(f"Snippet: {snippet}")

    return "\n".join(lines)


# ── LangSmith-traced LLM call ──────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="filter-agent-gemma4",
    tags=["filter-agent", "gemma4", "local"],
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
    OTel spans (filter_agent.filter / filter_agent.llm_call) are opened by
    the caller; this function handles only the Ollama network call.
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


# ── Per-batch filtering ────────────────────────────────────────────────────────

async def _filter_batch(
    batch: list[tuple[ScoredArticle, ResultConfidence]],
    constraints: list[str],
    *,
    model_id: str,
    prompt_version: str,
    system_prompt: str,
    user_template: str,
    run_id: str,
    tracer: Any,
) -> list[dict[str, Any]] | PipelineError:
    """
    Call Gemma 4 to evaluate one batch of articles against constraints.

    Returns a list of result dicts (one per article) or a PipelineError.
    Keys: article, confidence, keep, constraint_index (int|None),
    llm_confidence, note.
    """
    batch_payload = _build_batch_payload(batch, constraints)
    user_prompt = user_template.format(batch_payload=batch_payload)
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4

    with tracer.start_as_current_span("filter_agent.llm_call") as llm_span:
        llm_span.set_attribute("model_id", model_id)
        llm_span.set_attribute("prompt_version", prompt_version)
        llm_span.set_attribute("batch_size", len(batch))
        llm_span.set_attribute("token_estimate", token_estimate)
        # Never record constraint text or article content in span attributes

        try:
            ollama_resp = await _call_ollama(
                user_prompt,
                system_prompt,
                run_id,
                model_id=model_id,
            )
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
            error = PipelineError(
                error_code=code,
                severity=ErrorSeverity.FATAL,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message=(
                    f"Ollama {'model not available' if is_model_missing else 'API error'}: "
                    f"{model_id}"
                ),
                retry_hint=hint,
                context={"model_id": model_id, "status_code": status},
            )
            record_span_error(llm_span, error)
            log_pipeline_error(logger, error)
            return error

        except (ConnectionError, OSError, TimeoutError) as exc:
            error = PipelineError(
                error_code=OLLAMA_UNAVAILABLE,
                severity=ErrorSeverity.FATAL,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message="Ollama service is unreachable",
                retry_hint="Ensure Ollama is running at OLLAMA_BASE_URL",
                context={"error_type": type(exc).__name__},
            )
            record_span_error(llm_span, error)
            log_pipeline_error(logger, error)
            return error

        except Exception as exc:
            error = PipelineError(
                error_code=OLLAMA_UNAVAILABLE,
                severity=ErrorSeverity.FATAL,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message="Unexpected error calling Ollama",
                retry_hint="Check Ollama connectivity and model availability",
                context={"error_type": type(exc).__name__},
            )
            record_span_error(llm_span, error)
            log_pipeline_error(logger, error)
            return error

    # ── Parse JSON response ─────────────────────────────────────────────────
    try:
        raw_content: str = ollama_resp["message"]["content"]
        parsed: dict[str, Any] = json.loads(raw_content)
        llm_results: list[dict[str, Any]] = parsed["results"]
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        error = PipelineError(
            error_code=INVALID_LLM_RESPONSE,
            severity=ErrorSeverity.RETRYABLE,
            agent_id=_AGENT_ID,
            run_id=run_id,
            message="LLM returned invalid or unparseable JSON",
            retry_hint="Retry; Ollama format=json should produce valid JSON",
            context={"batch_size": len(batch), "error_type": type(exc).__name__},
        )
        log_pipeline_error(logger, error)
        return error

    # Build URL → LLM decision lookup
    llm_by_url: dict[str, dict[str, Any]] = {
        r["url"]: r for r in llm_results if isinstance(r.get("url"), str)
    }

    output: list[dict[str, Any]] = []
    for sa, conf in batch:
        url = sa.article.url
        llm = llm_by_url.get(url)
        if llm is None:
            # LLM did not return an entry for this URL — default to keep
            output.append({
                "article": sa,
                "confidence": conf,
                "keep": True,
                "constraint_index": None,
                "llm_confidence": 0.5,
                "note": "",
            })
            continue

        keep: bool = bool(llm.get("keep", True))
        constraint_idx: Optional[int] = llm.get("violated_constraint_index")
        llm_conf: float = max(0.0, min(1.0, float(llm.get("confidence", 0.5))))

        note = ""
        if not keep and conf == ResultConfidence.SNIPPET:
            note = "removed based on snippet only — limited context"

        output.append({
            "article": sa,
            "confidence": conf,
            "keep": keep,
            "constraint_index": constraint_idx,
            "llm_confidence": llm_conf,
            "note": note,
        })

    return output


# ── LangSmith-traced filter run ────────────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="filter_agent",
    tags=["agent:filter_agent"],
)
async def _traced_run(
    inp: FilterAgentInput,
    *,
    model_id: str,
    prompt_version: str,
    tracer: Any,
) -> FilterAgentOutput:
    """LangSmith-traced body. inp is captured as run input for tracing."""
    prompt_data = load_prompt("filter_agent", prompt_version)
    system_prompt = get_system_prompt("filter_agent", prompt_version)
    user_template: str = prompt_data.get("user_template", "")

    # ── Validate constraints ───────────────────────────────────────────────
    valid_constraints, _rejected_count = _sanitize_constraints(
        inp.constraints, run_id=inp.run_id
    )

    # ── Build confidence lookup from search_results ────────────────────────
    confidence_lookup: dict[str, ResultConfidence] = {
        sr.url: sr.confidence for sr in inp.search_results
    }

    # ── Separate INJECTED articles (auto-remove, no LLM call) ─────────────
    injected_removals: list[RemovalReason] = []
    injected_removed: list[ScoredArticle] = []
    candidates: list[tuple[ScoredArticle, ResultConfidence]] = []

    for sa in inp.scored_articles:
        conf = confidence_lookup.get(sa.article.url, ResultConfidence.FULL)
        if conf == ResultConfidence.INJECTED:
            injected_removed.append(sa)
            injected_removals.append(RemovalReason(
                url=sa.article.url,
                constraint_violated=-1,  # -1 = auto-removed, not a constraint violation
                confidence=1.0,
                note="auto-removed: injected content detected",
            ))
        else:
            candidates.append((sa, conf))

    # ── All results injected → ALL_FILTERED SKIP ──────────────────────────
    if injected_removed and not candidates:
        error = PipelineError(
            error_code=ALL_FILTERED,
            severity=ErrorSeverity.SKIP,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="All results were injected content and auto-removed",
            retry_hint="Source quality issue — all results had injection flag",
            context={
                "total_input": len(inp.scored_articles),
                "injected_count": len(injected_removed),
            },
        )
        log_pipeline_error(logger, error)
        return FilterAgentOutput(
            run_id=inp.run_id,
            success=False,
            error=error,
            kept_results=[],
            removed_results=injected_removed,
            removal_reasons=injected_removals,
            filtered_articles=[],
            dropped_count=len(injected_removed),
        )

    # ── No constraints → semantic pass-through ─────────────────────────────
    if not valid_constraints:
        kept = [sa for sa, _ in candidates]
        dq_error = _validator.validate_filtered_results(
            filtered_count=len(kept),
            original_count=len(inp.scored_articles),
            run_id=inp.run_id,
        )
        if dq_error:
            log_pipeline_error(logger, dq_error)
            return FilterAgentOutput(
                run_id=inp.run_id,
                success=False,
                error=dq_error,
                kept_results=[],
                removed_results=injected_removed,
                removal_reasons=injected_removals,
                filtered_articles=[],
                dropped_count=len(inp.scored_articles),
            )

        logger.info(
            "filter_agent_no_constraints",
            extra={
                "run_id": inp.run_id,
                "kept_count": len(kept),
                "injected_removed": len(injected_removed),
            },
        )
        return FilterAgentOutput(
            run_id=inp.run_id,
            success=True,
            kept_results=kept,
            removed_results=injected_removed,
            removal_reasons=injected_removals,
            filtered_articles=kept,
            dropped_count=len(injected_removed),
        )

    # ── Batch process with Gemma 4 ─────────────────────────────────────────
    batch_size = _cfg_batch_size()
    batches: list[list[tuple[ScoredArticle, ResultConfidence]]] = [
        candidates[i : i + batch_size]
        for i in range(0, len(candidates), batch_size)
    ]

    kept_results: list[ScoredArticle] = []
    removed_results: list[ScoredArticle] = list(injected_removed)
    removal_reasons: list[RemovalReason] = list(injected_removals)
    warnings: list[str] = []

    with tracer.start_as_current_span("filter_agent.filter") as filter_span:
        filter_span.set_attribute("run_id", inp.run_id)
        filter_span.set_attribute("input_count", len(inp.scored_articles))
        filter_span.set_attribute("constraints_count", len(valid_constraints))
        # Never record constraint text or article content in spans

        for batch_idx, batch in enumerate(batches):
            result = await _filter_batch(
                batch,
                valid_constraints,
                model_id=model_id,
                prompt_version=prompt_version,
                system_prompt=system_prompt,
                user_template=user_template,
                run_id=inp.run_id,
                tracer=tracer,
            )

            if isinstance(result, PipelineError):
                if result.severity == ErrorSeverity.FATAL:
                    record_span_error(filter_span, result)
                    return FilterAgentOutput(
                        run_id=inp.run_id,
                        success=False,
                        error=result,
                        kept_results=[],
                        removed_results=removed_results,
                        removal_reasons=removal_reasons,
                        filtered_articles=[],
                        dropped_count=len(inp.scored_articles),
                    )
                # RETRYABLE — skip batch, treat articles as kept to not lose them
                warnings.append(f"batch_{batch_idx}_skipped:{result.error_code}")
                for sa, _ in batch:
                    kept_results.append(sa)
                continue

            for item in result:
                sa: ScoredArticle = item["article"]
                if item["keep"]:
                    kept_results.append(sa)
                else:
                    removed_results.append(sa)
                    constraint_idx = item["constraint_index"]
                    removal_reasons.append(RemovalReason(
                        url=sa.article.url,
                        constraint_violated=(
                            constraint_idx if constraint_idx is not None else 0
                        ),
                        confidence=item["llm_confidence"],
                        note=item["note"],
                    ))

        filter_span.set_attribute("kept_count", len(kept_results))
        filter_span.set_attribute("removed_count", len(removed_results))

    # ── High-removal rate warning (> 80% semantic filter) ─────────────────
    total_candidates = len(candidates)
    semantic_removed = len(removed_results) - len(injected_removed)
    if total_candidates > 0 and semantic_removed / total_candidates > 0.8:
        logger.warning(
            "filter_agent_high_removal_rate",
            extra={
                "run_id": inp.run_id,
                "total_candidates": total_candidates,
                "semantic_removed": semantic_removed,
                "constraints_count": len(valid_constraints),
            },
        )
        warnings.append("high_removal_rate:DEGRADED")

    # ── All filtered → ALL_FILTERED SKIP ──────────────────────────────────
    dq_error = _validator.validate_filtered_results(
        filtered_count=len(kept_results),
        original_count=len(inp.scored_articles),
        run_id=inp.run_id,
    )
    if dq_error:
        log_pipeline_error(logger, dq_error)
        return FilterAgentOutput(
            run_id=inp.run_id,
            success=False,
            error=dq_error,
            kept_results=[],
            removed_results=removed_results,
            removal_reasons=removal_reasons,
            filtered_articles=[],
            dropped_count=len(inp.scored_articles),
            warnings=warnings,
        )

    logger.info(
        "filter_agent_complete",
        extra={
            "run_id": inp.run_id,
            "input_count": len(inp.scored_articles),
            "kept_count": len(kept_results),
            "removed_count": len(removed_results),
            "constraints_used": len(valid_constraints),
            # Never log constraint text
        },
    )

    return FilterAgentOutput(
        run_id=inp.run_id,
        success=True,
        kept_results=kept_results,
        removed_results=removed_results,
        removal_reasons=removal_reasons,
        filtered_articles=kept_results,
        dropped_count=len(removed_results),
        warnings=warnings,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: FilterAgentInput,
    *,
    tracer: Any = None,
    model_id: str | None = None,
) -> FilterAgentOutput:
    """
    Public entry point. ``model_id`` is optional — main.py injects it from
    AgentConfig; callers that omit it get MODEL_OVERRIDE or the default model.
    ``tracer`` is optional; pass a test-scoped tracer to capture spans without
    touching the global OTel provider.
    """
    load_dotenv()
    setup_telemetry("news-mas-filter-agent")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("filter-agent")

    prompt_version = _PROMPT_VERSION
    _override = os.getenv("MODEL_OVERRIDE", "").strip()
    effective_model = (
        model_id
        or (_override if _override and not _override.startswith("#") else None)
        or _DEFAULT_MODEL
    )

    make_run_config("filter_agent", prompt_version, run_id=inp.run_id)
    logger.info(
        "filter_agent_start",
        extra={
            "run_id": inp.run_id,
            "article_count": len(inp.scored_articles),
            "constraints_count": len(inp.constraints),
            "model_id": effective_model,
            "langsmith_active": langsmith_active,
            # Never log constraint text
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
            return FilterAgentOutput(
                run_id=inp.run_id,
                success=False,
                error=error,
                kept_results=[],
                removed_results=[],
                removal_reasons=[],
                filtered_articles=[],
                dropped_count=0,
            )

    return await _traced_run(
        inp,
        model_id=effective_model,
        prompt_version=prompt_version,
        tracer=active_tracer,
    )
