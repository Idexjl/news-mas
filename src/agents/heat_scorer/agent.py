"""
HeatScorer agent — scores news articles by heat using Gemma 4 via Ollama.

Scoring signals (all computed by the LLM):
  volume_signal:      breadth of independent source coverage
  velocity_signal:    recency and speed of coverage accumulation
  novelty_signal:     genuinely new development vs. recurring background
  significance_signal: durable real-world impact vs. ephemeral hot take

Injection defence:
  - Hardened system preamble on every call (loaded from prompt YAML)
  - Only article title + snippet (first 500 chars) sent to LLM — never full content
  - OTel spans carry zero article content (titles, snippets, reasoning excluded)

OTel spans emitted per article:
  heat_scorer.score     (parent) → run_id, result_count, heat_score, model_id, article_index
  heat_scorer.llm_call  (child)  → model_id, prompt_version, token_estimate
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

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
    NO_RESULTS,
    OLLAMA_UNAVAILABLE,
    SCORE_OUT_OF_RANGE,
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
from src.common.schemas import CandidateConfidence, HeatScorerInput, HeatScorerOutput, RawArticle, ScoredArticle

_DEFAULT_MODEL = "gemma4:e4b"
_AGENT_ID = "heat-scorer"
_PROMPT_VERSION = "v1.0"
_REQUIRED_CAPABILITY = "score.heat"
_SNIPPET_LENGTH = 500  # chars — never pass full article content to LLM

_validator = DataQualityValidator(agent_id=_AGENT_ID)
logger = get_logger(__name__)


# ── AAP token validation ───────────────────────────────────────────────────────

def _validate_aap_token(token: str) -> dict[str, Any]:
    """Validate AAP token and assert score.heat capability. Logs actor claims only."""
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


# ── LangSmith-traced LLM calls ────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="heat-scorer-gemma4",
    tags=["heat-scorer", "gemma4", "local"],
    metadata={"model": "gemma4:e4b"},
)
async def _call_ollama(
    prompt: str,
    system: str,
    topic_id: str,
    run_id: str,
    *,
    model_id: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    LangSmith-traced Ollama inference call.

    ``prompt`` (user turn) and ``system`` are captured as LangSmith run inputs
    so every trace shows the full prompt/completion pair. ``topic_id`` and
    ``run_id`` appear as inputs for cross-referencing with OTel traces.
    ``model_id`` is a kwarg so override values are visible in the trace.

    OTel spans (heat_scorer.score / heat_scorer.llm_call) are opened by the
    caller; this function is responsible only for the Ollama network call.
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


@_ls_traceable(
    run_type="llm",
    name="heat-scorer-anthropic",
    tags=["heat-scorer", "anthropic", "cloud"],
    metadata={"provider": "anthropic"},
)
async def _call_anthropic(
    prompt: str,
    system: str,
    topic_id: str,
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


# ── Per-article scoring ────────────────────────────────────────────────────────

async def _score_one_article(
    article: RawArticle,
    article_index: int,
    *,
    model_id: str,
    provider: str,
    prompt_version: str,
    system_prompt: str,
    user_template: str,
    run_id: str,
    result_count: int,
    tracer: Any,
) -> ScoredArticle | PipelineError:
    """
    Score one article. Returns ScoredArticle on success, PipelineError on failure.

    FATAL errors (Ollama unavailable, model not found) abort the entire run.
    RETRYABLE errors (invalid JSON, score out of range) skip the article.
    """
    # Snippet only — never send full article content to the LLM
    snippet = article.content[:_SNIPPET_LENGTH]
    published_str = (
        article.published_at.isoformat() if article.published_at else "unknown"
    )
    user_prompt = user_template.format(
        title=article.title,
        source=article.source,
        published_at=published_str,
        snippet=snippet,
    )
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4

    with tracer.start_as_current_span("heat_scorer.score") as score_span:
        # Attributes: structural metadata only — no titles, snippets, or content
        score_span.set_attribute("run_id", run_id)
        score_span.set_attribute("model_id", model_id)
        score_span.set_attribute("result_count", result_count)
        score_span.set_attribute("article_index", article_index)
        score_span.set_attribute("topic_id", "")  # not in HeatScorerInput schema

        with tracer.start_as_current_span("heat_scorer.llm_call") as llm_span:
            llm_span.set_attribute("model_id", model_id)
            llm_span.set_attribute("prompt_version", prompt_version)
            llm_span.set_attribute("token_estimate", token_estimate)

            try:
                if provider == "anthropic":
                    ollama_resp = await _call_anthropic(
                        user_prompt, system_prompt, "", run_id, model_id=model_id,
                    )
                else:
                    ollama_resp = await _call_ollama(
                        user_prompt, system_prompt, "", run_id, model_id=model_id,
                    )
            except _OllamaResponseError as exc:
                status = getattr(exc, "status_code", -1)
                msg_lower = str(exc).lower()
                is_model_missing = (
                    status == 404
                    or ("not found" in msg_lower and "model" in msg_lower)
                    or ("pull" in msg_lower and "model" in msg_lower)
                )
                if is_model_missing:
                    error = PipelineError(
                        error_code=MODEL_NOT_FOUND,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=run_id,
                        message=f"Ollama model not available: {model_id}",
                        retry_hint=f"Run 'ollama pull {model_id}' to download the model",
                        context={"model_id": model_id, "status_code": status},
                    )
                else:
                    error = PipelineError(
                        error_code=OLLAMA_UNAVAILABLE,
                        severity=ErrorSeverity.FATAL,
                        agent_id=_AGENT_ID,
                        run_id=run_id,
                        message="Ollama API error during model inference",
                        retry_hint="Check Ollama server status at OLLAMA_BASE_URL",
                        context={"model_id": model_id, "status_code": status},
                    )
                record_span_error(llm_span, error)
                record_span_error(score_span, error)
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
                record_span_error(score_span, error)
                log_pipeline_error(logger, error)
                return error

            except Exception as exc:
                error = PipelineError(
                    error_code=OLLAMA_UNAVAILABLE,
                    severity=ErrorSeverity.FATAL,
                    agent_id=_AGENT_ID,
                    run_id=run_id,
                    message="Unexpected error calling LLM",
                    retry_hint="Check LLM backend connectivity and model availability",
                    context={"error_type": type(exc).__name__, "provider": provider},
                )
                record_span_error(llm_span, error)
                record_span_error(score_span, error)
                log_pipeline_error(logger, error)
                return error

        # ── Parse JSON response ─────────────────────────────────────────────
        try:
            raw_content: str = ollama_resp["message"]["content"]
            parsed: dict[str, Any] = json.loads(raw_content)
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            error = PipelineError(
                error_code=INVALID_LLM_RESPONSE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message="LLM returned invalid or unparseable JSON",
                retry_hint="Retry; Ollama format=json should produce valid JSON",
                context={
                    "article_index": article_index,
                    "error_type": type(exc).__name__,
                },
            )
            record_span_error(score_span, error)
            log_pipeline_error(logger, error)
            return error

        # ── Validate required fields ────────────────────────────────────────
        required_fields = {
            "heat_score", "reasoning",
            "volume_signal", "velocity_signal",
            "novelty_signal", "significance_signal",
        }
        missing = required_fields - parsed.keys()
        if missing:
            error = PipelineError(
                error_code=INVALID_LLM_RESPONSE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message="LLM response missing required schema fields",
                retry_hint="Retry; response must match prompt output schema",
                context={
                    "article_index": article_index,
                    "missing_field_count": len(missing),
                },
            )
            record_span_error(score_span, error)
            log_pipeline_error(logger, error)
            return error

        # ── Parse and range-check heat_score ───────────────────────────────
        try:
            heat_score = float(parsed["heat_score"])
        except (ValueError, TypeError):
            error = PipelineError(
                error_code=INVALID_LLM_RESPONSE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message="heat_score field is not a valid float",
                retry_hint="Retry",
                context={"article_index": article_index},
            )
            record_span_error(score_span, error)
            log_pipeline_error(logger, error)
            return error

        if not (0.0 <= heat_score <= 1.0):
            error = PipelineError(
                error_code=SCORE_OUT_OF_RANGE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=run_id,
                message=(
                    f"heat_score {heat_score:.4f} is outside valid range [0.0, 1.0]"
                ),
                retry_hint="Retry; prompt instructs score must be in [0.0, 1.0]",
                context={"article_index": article_index, "heat_score": heat_score},
            )
            record_span_error(score_span, error)
            log_pipeline_error(logger, error)
            return error

        # Clamp to guard against floating-point edge cases at the boundaries
        heat_score = max(0.0, min(1.0, heat_score))

        # Set output attribute AFTER validation — no content, only the score
        score_span.set_attribute("heat_score", heat_score)

        return ScoredArticle(
            article=article,
            heat_score=heat_score,
            reasoning=parsed.get("reasoning", ""),
        )


# ── LangSmith-traced batch scorer ──────────────────────────────────────────────

@_ls_traceable(
    run_type="chain",
    name="heat_scorer",
    tags=["agent:heat_scorer"],
)
async def _traced_run(
    inp: HeatScorerInput,
    *,
    model_id: str,
    provider: str,
    prompt_version: str,
    tracer: Any,
) -> HeatScorerOutput:
    """
    LangSmith-traced body. ``inp`` is captured as the run input so run_id and
    article count appear in every LangSmith trace automatically. ``model_id``
    and ``prompt_version`` appear as run metadata for filtering.
    """
    prompt_data = load_prompt("heat_scorer", prompt_version)
    system_prompt = get_system_prompt("heat_scorer", prompt_version)
    user_template: str = prompt_data.get("user_template", "")

    # Validate input not empty
    dq_error = _validator.validate_search_results(
        inp.articles,
        min_results=1,
        topic_id="",
        run_id=inp.run_id,
    )
    if dq_error is not None:
        log_pipeline_error(logger, dq_error)
        return HeatScorerOutput(
            run_id=inp.run_id,
            success=False,
            error=dq_error,
            scored_articles=[],
        )

    scored_articles: list[ScoredArticle] = []
    warnings: list[str] = []
    result_count = len(inp.articles)

    for i, article in enumerate(inp.articles):
        # Skip articles where injection was detected — content was discarded.
        if article.confidence == ResultConfidence.INJECTED:
            warnings.append(f"article_{i}_skipped:injected")
            continue

        result = await _score_one_article(
            article,
            i,
            model_id=model_id,
            provider=provider,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            user_template=user_template,
            run_id=inp.run_id,
            result_count=result_count,
            tracer=tracer,
        )

        if isinstance(result, PipelineError):
            if result.severity == ErrorSeverity.FATAL:
                # Abort entire run — infrastructure is down
                return HeatScorerOutput(
                    run_id=inp.run_id,
                    success=False,
                    error=result,
                    scored_articles=[],
                )
            # RETRYABLE / SKIP — skip this article, continue with the rest
            warnings.append(f"article_{i}_skipped:{result.error_code}")
        else:
            scored_articles.append(result)

    avg_heat = (
        sum(sa.heat_score for sa in scored_articles) / len(scored_articles)
        if scored_articles else 0.0
    )
    candidate_confidence = CandidateConfidence.from_results(
        inp.articles,
        heat_score=avg_heat,
    )

    logger.info(
        "heat_scorer_complete",
        extra={
            "run_id": inp.run_id,
            "articles_scored": len(scored_articles),
            "articles_skipped": len(warnings),
            "model_id": model_id,
            "overall_confidence": candidate_confidence.overall_confidence,
            "confidence_ratio": candidate_confidence.confidence_ratio,
        },
    )

    return HeatScorerOutput(
        run_id=inp.run_id,
        success=True,
        scored_articles=scored_articles,
        warnings=warnings,
        candidate_confidence=candidate_confidence,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: HeatScorerInput,
    *,
    tracer: Any = None,
    model_id: str | None = None,
    model_provider: str | None = None,
) -> HeatScorerOutput:
    """
    Public entry point. ``model_id`` and ``model_provider`` are optional —
    main.py injects them from AgentConfig. MODEL_OVERRIDE is applied via
    resolve_model(); summarizer and reviewer are exempt.
    ``tracer`` is optional; pass a test-scoped tracer to capture spans without
    touching the global OTel provider.
    """
    load_dotenv()
    setup_telemetry("news-mas-heat-scorer")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("heat-scorer")

    prompt_version = _PROMPT_VERSION
    provider, effective_model = resolve_model(
        model_id or _DEFAULT_MODEL,
        model_provider or "ollama",
        _AGENT_ID,
    )

    run_cfg = make_run_config("heat_scorer", prompt_version, run_id=inp.run_id)
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    logger.info(
        "heat_scorer_start",
        extra={
            "run_id": inp.run_id,
            "article_count": len(inp.articles),
            "model_id": effective_model,
            "langsmith_active": langsmith_active,
            "otel_endpoint": otel_endpoint,
            "langsmith_run_name": run_cfg["run_name"],
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
            return HeatScorerOutput(
                run_id=inp.run_id,
                success=False,
                error=error,
                scored_articles=[],
            )

    return await _traced_run(
        inp,
        model_id=effective_model,
        provider=provider,
        prompt_version=prompt_version,
        tracer=active_tracer,
    )
