"""
Summarizer Agent — produces user-facing summaries with cited sources.

Model: claude-sonnet-4-6 (hardcoded; MODEL_OVERRIDE never applies here).
Context strategy: direct for ≤TOKEN_BUDGET_SOURCES tokens; map-reduce above.
PHI: inputs already scrubbed by SearchWorker; output is re-checked belt-and-suspenders.
Citations: every URL validated against input source URLs — hallucinated URLs are RETRYABLE.

OTel spans emitted:
  summarizer.summarize        (parent) → strategy, source_count, token_estimate,
                                          retry_count, confidence, citation_count
  summarizer.llm_call         (child)  → model_id, prompt_version, token_estimate, retry_count
  summarizer.map_reduce       (child)  → source_count, mini_summary_count
  summarizer.citation_validation       → citations_checked, invalid_count, valid_count
  summarizer.phi_check        (child)  → phi_detected, entities_scrubbed_count

Security invariants:
  - Summary content NEVER in spans, logs, or error messages
  - Source content NEVER in spans or logs
  - Topic text NEVER in spans or logs
"""
from __future__ import annotations

import asyncio
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
    import anthropic as _anthropic
    _AnthropicRateLimitError = _anthropic.RateLimitError
    _AnthropicAPIError = _anthropic.APIError
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _AnthropicRateLimitError = Exception  # type: ignore[assignment,misc]
    _AnthropicAPIError = Exception  # type: ignore[assignment,misc]

from src.auth.token_service import validate_token
from src.common.data_quality import DataQualityValidator
from src.common.error_codes import (
    AUTH_FAILED,
    CITATION_INVALID,
    INVALID_LLM_RESPONSE,
    PHI_DETECTED,
    SUMMARY_TOO_SHORT,
)
from src.common.observability import (
    configure_langsmith,
    get_logger,
    get_tracer,
    log_pipeline_error,
    record_span_error,
    setup_telemetry,
)
from src.common.pii_scrubber import detect_phi_in_output, scrub_text
from src.common.pipeline_errors import AgentResult, ErrorSeverity, PipelineError  # noqa: F401
from src.common.prompt_loader import get_system_prompt, load_prompt, make_run_config
from src.common.schemas import (
    Citation,
    RawArticle,
    SummarizerInput,
    SummarizerOutput,
)

_AGENT_ID = "summarizer"
_MODEL_ID = "claude-sonnet-4-6"   # hardcoded; never overridden by MODEL_OVERRIDE
_PROMPT_VERSION = "v1.0"
_MINI_PROMPT_NAME = f"mini_{_PROMPT_VERSION}"
_REQUIRED_CAPABILITY = "summarize.content"
_MAX_API_RETRIES = 2
_MIN_SUMMARY_WORDS = 100

_validator = DataQualityValidator(agent_id=_AGENT_ID)
logger = get_logger(__name__)

TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET_SOURCES", "16000"))


# ── AAP token validation ────────────────────────────────────────────────────────

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


# ── Source helpers ──────────────────────────────────────────────────────────────

def _get_sources(inp: SummarizerInput) -> list[RawArticle]:
    """Return sources list, falling back to inp.article as the single source."""
    if inp.sources:
        return list(inp.sources)
    return [inp.article]


def _estimate_tokens(sources: list[RawArticle]) -> int:
    return sum(len(s.content or "") // 4 for s in sources)


def _format_sources_block(sources: list[RawArticle]) -> str:
    parts: list[str] = []
    for idx, src in enumerate(sources, start=1):
        title = src.title or "(untitled)"
        url = src.url or "(no url)"
        content = (src.content or "").strip()
        parts.append(
            f"[{idx}] Title: {title}\n"
            f"    URL:   {url}\n\n"
            f"{content}\n"
            f"{'-' * 40}"
        )
    return "\n\n".join(parts)


def _build_feedback_block(reviewer_feedback: Optional[str]) -> str:
    if not reviewer_feedback:
        return ""
    return (
        f"IMPORTANT — Previous attempt was rejected. Issues found:\n"
        f"{reviewer_feedback}\n"
        f"Please address these specific issues in your summary.\n\n"
    )


# ── Anthropic LLM call ─────────────────────────────────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="summarizer-claude-sonnet",
    tags=["summarizer", "anthropic", "claude-sonnet"],
    metadata={"model": _MODEL_ID},
)
async def _call_claude(
    user_prompt: str,
    system_prompt: str,
    *,
    run_id: str,
    max_tokens: int = 2048,
) -> str:
    """
    LangSmith-traced Claude Sonnet call. Returns stripped response text.
    Retries up to _MAX_API_RETRIES for transient API and rate-limit errors.
    """
    if _anthropic is None:
        raise RuntimeError("anthropic package is not installed")

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
                await asyncio.sleep(2 ** attempt)

        except _AnthropicAPIError as exc:
            last_exc = exc
            if attempt < _MAX_API_RETRIES:
                await asyncio.sleep(1)

    raise last_exc or RuntimeError("Anthropic API failed after retries")


# ── Mini-summarizer (map-reduce per-source step) ───────────────────────────────

@_ls_traceable(
    run_type="llm",
    name="mini-summarizer-claude-sonnet",
    tags=["summarizer", "mini", "anthropic", "claude-sonnet", "strategy:map_reduce"],
    metadata={"model": _MODEL_ID, "step": "map_reduce_mini"},
)
async def _call_claude_mini(
    user_prompt: str,
    system_prompt: str,
    *,
    run_id: str,
    source_url: str,
) -> dict[str, Any]:
    """
    LangSmith-traced mini-summarizer call for one source.
    Preserves source_url exactly so citations can be grounded downstream.
    """
    text = await _call_claude(user_prompt, system_prompt, run_id=run_id, max_tokens=512)
    try:
        parsed: dict[str, Any] = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = {"mini_summary": text[:600], "key_points": []}
    # Always pin source_url to the input — never trust what the model returns
    parsed["source_url"] = source_url
    return parsed


async def _mini_summarize_one(
    source: RawArticle,
    topic: str,
    system_prompt: str,
    user_template: str,
    run_id: str,
) -> dict[str, Any]:
    content = (source.content or "").strip()[:8000]  # hard per-source cap
    user_prompt = user_template.format(
        topic=topic,
        source_url=source.url or "",
        title=source.title or "(untitled)",
        content=content,
    )
    return await _call_claude_mini(
        user_prompt,
        system_prompt,
        run_id=run_id,
        source_url=source.url or "",
    )


# ── Output parsing and validation ─────────────────────────────────────────────

def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_summary_length(
    parsed: dict[str, Any], run_id: str, topic_id: str
) -> PipelineError | None:
    summary = parsed.get("summary", "")
    word_count = len(summary.split())
    if word_count < _MIN_SUMMARY_WORDS:
        return PipelineError(
            error_code=SUMMARY_TOO_SHORT,
            severity=ErrorSeverity.RETRYABLE,
            agent_id=_AGENT_ID,
            run_id=run_id,
            topic_id=topic_id,
            message=f"Summary too short: {word_count} words (minimum {_MIN_SUMMARY_WORDS})",
            retry_hint="Re-prompt with explicit word count requirement",
            context={"word_count": word_count, "min_words": _MIN_SUMMARY_WORDS},
        )
    return None


def _check_and_scrub_phi(
    summary: str, key_points: list[str]
) -> tuple[str, list[str], int]:
    """
    Belt-and-suspenders PHI check on LLM output.

    Uses detect_phi_in_output (restricted entity set, high confidence threshold)
    so that researcher names, institutions, and dates in news prose do not
    generate false positives. Only genuine PHI identifiers (SSNs, NHS numbers,
    medical licences, etc.) trigger rejection.

    Returns (scrubbed_summary, scrubbed_points, detection_count).
    Content is never logged.
    """
    combined = summary + " " + " ".join(key_points)
    detections = detect_phi_in_output(combined)
    if not detections:
        return summary, key_points, 0
    return scrub_text(summary), [scrub_text(p) for p in key_points], len(detections)


# ── Direct summarization (Option A — single LLM call) ─────────────────────────

@_ls_traceable(
    run_type="chain",
    name="summarizer-direct",
    tags=["summarizer", "strategy:direct"],
)
async def _direct_summarize(
    inp: SummarizerInput,
    sources: list[RawArticle],
    system_prompt: str,
    prompt_data: dict[str, Any],
    tracer: Any,
) -> tuple[dict[str, Any], PipelineError | None]:
    """
    Direct path: pass all sources in a single Claude Sonnet call.
    Returns (parsed_dict, error). error is set on JSON parse failure.
    """
    topic = inp.topic_text or inp.article.title or "news"
    sources_block = _format_sources_block(sources)
    feedback_block = _build_feedback_block(inp.reviewer_feedback)
    user_template: str = prompt_data.get("user_template", "")
    user_prompt = user_template.format(
        topic=topic,
        source_count=len(sources),
        sources_block=sources_block,
        reviewer_feedback_block=feedback_block,
    )
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4
    retry_count = inp.retry_count or 0

    with tracer.start_as_current_span("summarizer.llm_call") as llm_span:
        llm_span.set_attribute("model_id", _MODEL_ID)
        llm_span.set_attribute("prompt_version", _PROMPT_VERSION)
        llm_span.set_attribute("token_estimate", token_estimate)
        llm_span.set_attribute("retry_count", retry_count)

        raw = await _call_claude(user_prompt, system_prompt, run_id=inp.run_id)

    parsed = _parse_json(raw)
    if parsed is None:
        return {}, PipelineError(
            error_code=INVALID_LLM_RESPONSE,
            severity=ErrorSeverity.RETRYABLE,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="Claude returned unparseable JSON in direct summarization",
            retry_hint="Retry; reinforce JSON-only output in system prompt",
            context={"strategy": "direct", "retry_count": retry_count},
        )
    return parsed, None


# ── Map-reduce summarization (Option B — mini then synthesize) ────────────────

@_ls_traceable(
    run_type="chain",
    name="summarizer-map-reduce",
    tags=["summarizer", "strategy:map_reduce"],
)
async def _map_reduce_summarize(
    inp: SummarizerInput,
    sources: list[RawArticle],
    system_prompt: str,
    prompt_data: dict[str, Any],
    tracer: Any,
) -> tuple[dict[str, Any], PipelineError | None]:
    """
    Map-reduce path: mini-summarize each source concurrently, then synthesize.
    Returns (parsed_dict, error). error is set on JSON parse failure.
    """
    mini_prompt_data = load_prompt("summarizer", _MINI_PROMPT_NAME)
    mini_system = get_system_prompt("summarizer", _MINI_PROMPT_NAME)
    mini_user_template: str = mini_prompt_data.get("user_template", "")
    topic = inp.topic_text or inp.article.title or "news"

    with tracer.start_as_current_span("summarizer.map_reduce") as mr_span:
        mr_span.set_attribute("source_count", len(sources))
        tasks = [
            _mini_summarize_one(src, topic, mini_system, mini_user_template, inp.run_id)
            for src in sources
        ]
        mini_results: list[dict[str, Any]] = await asyncio.gather(*tasks)
        mr_span.set_attribute("mini_summary_count", len(mini_results))

    # Build synthesis input from mini-summaries (original URLs preserved)
    mini_block_parts: list[str] = []
    for idx, mini in enumerate(mini_results, start=1):
        ms = mini.get("mini_summary", "")
        url = mini.get("source_url", "")
        kps = mini.get("key_points", [])
        kp_str = "\n".join(f"  - {kp}" for kp in kps) if kps else "  (none)"
        mini_block_parts.append(
            f"[{idx}] URL: {url}\n"
            f"Mini-summary: {ms}\n"
            f"Key points:\n{kp_str}\n"
            f"{'-' * 40}"
        )
    synthesis_block = "\n\n".join(mini_block_parts)

    feedback_block = _build_feedback_block(inp.reviewer_feedback)
    user_template: str = prompt_data.get("user_template", "")
    user_prompt = user_template.format(
        topic=topic,
        source_count=len(sources),
        sources_block=synthesis_block,
        reviewer_feedback_block=feedback_block,
    )
    token_estimate = (len(system_prompt) + len(user_prompt)) // 4
    retry_count = inp.retry_count or 0

    with tracer.start_as_current_span("summarizer.llm_call") as llm_span:
        llm_span.set_attribute("model_id", _MODEL_ID)
        llm_span.set_attribute("prompt_version", _PROMPT_VERSION)
        llm_span.set_attribute("token_estimate", token_estimate)
        llm_span.set_attribute("retry_count", retry_count)

        raw = await _call_claude(user_prompt, system_prompt, run_id=inp.run_id)

    parsed = _parse_json(raw)
    if parsed is None:
        return {}, PipelineError(
            error_code=INVALID_LLM_RESPONSE,
            severity=ErrorSeverity.RETRYABLE,
            agent_id=_AGENT_ID,
            run_id=inp.run_id,
            message="Claude returned unparseable JSON in map-reduce synthesis",
            retry_hint="Retry; reinforce JSON-only output in system prompt",
            context={"strategy": "map_reduce", "retry_count": retry_count},
        )
    return parsed, None


# ── LangSmith-traced core run ──────────────────────────────────────────────────

async def _core_run(
    inp: SummarizerInput,
    *,
    tracer: Any,
) -> SummarizerOutput:
    """Core summarization logic, shared between initial and retry trace wrappers."""
    prompt_data = load_prompt("summarizer", _PROMPT_VERSION)
    system_prompt = get_system_prompt("summarizer", _PROMPT_VERSION)

    sources = _get_sources(inp)
    token_estimate = _estimate_tokens(sources)
    strategy: Literal["direct", "map_reduce"] = (
        "direct" if token_estimate <= TOKEN_BUDGET else "map_reduce"
    )
    source_urls = {s.url for s in sources if s.url}
    retry_count = inp.retry_count or 0
    topic_id = inp.article.url or ""

    with tracer.start_as_current_span("summarizer.summarize") as parent_span:
        parent_span.set_attribute("topic_id", topic_id)
        parent_span.set_attribute("run_id", inp.run_id)
        parent_span.set_attribute("summarizer.strategy", strategy)
        parent_span.set_attribute("summarizer.source_count", len(sources))
        parent_span.set_attribute("summarizer.token_estimate", token_estimate)
        parent_span.set_attribute("summarizer.retry_count", retry_count)
        # Never record topic_text, article content, or reviewer_feedback in spans

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            if strategy == "direct":
                parsed, llm_error = await _direct_summarize(
                    inp, sources, system_prompt, prompt_data, tracer
                )
            else:
                parsed, llm_error = await _map_reduce_summarize(
                    inp, sources, system_prompt, prompt_data, tracer
                )
        except Exception as exc:
            llm_error = PipelineError(
                error_code=INVALID_LLM_RESPONSE,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                message="Anthropic API error during summarization",
                retry_hint="Retry; check ANTHROPIC_API_KEY and network connectivity",
                context={"error_type": type(exc).__name__, "strategy": strategy,
                         "retry_count": retry_count},
            )
            parsed = {}

        if llm_error:
            record_span_error(parent_span, llm_error)
            log_pipeline_error(logger, llm_error)
            return SummarizerOutput(run_id=inp.run_id, success=False, error=llm_error)

        # ── Citation validation ───────────────────────────────────────────────
        raw_citations: list[dict] = parsed.get("citations", [])
        citation_urls = [c.get("url", "") for c in raw_citations]

        with tracer.start_as_current_span("summarizer.citation_validation") as cv_span:
            invalid_count = sum(1 for u in citation_urls if u not in source_urls)
            valid_count = len(citation_urls) - invalid_count
            cv_span.set_attribute("citations_checked", len(citation_urls))
            cv_span.set_attribute("invalid_count", invalid_count)
            cv_span.set_attribute("valid_count", valid_count)

        if invalid_count > 0:
            cite_error = PipelineError(
                error_code=CITATION_INVALID,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                topic_id=topic_id,
                message=(
                    f"{invalid_count} of {len(citation_urls)} citation(s) "
                    "do not match any provided source URL"
                ),
                retry_hint=(
                    "Re-run with explicit instruction to cite only the provided URLs"
                ),
                context={
                    "total_citations": len(citation_urls),
                    "invalid_count": invalid_count,
                    "source_count": len(source_urls),
                    "retry_count": retry_count,
                },
            )
            record_span_error(parent_span, cite_error)
            log_pipeline_error(logger, cite_error)
            return SummarizerOutput(run_id=inp.run_id, success=False, error=cite_error)

        # ── Summary length validation ─────────────────────────────────────────
        len_error = _validate_summary_length(parsed, inp.run_id, topic_id)
        if len_error:
            record_span_error(parent_span, len_error)
            log_pipeline_error(logger, len_error)
            return SummarizerOutput(run_id=inp.run_id, success=False, error=len_error)

        # ── PHI check on output (belt-and-suspenders) ─────────────────────────
        summary_text = parsed.get("summary", "")
        key_points: list[str] = parsed.get("key_points", [])

        with tracer.start_as_current_span("summarizer.phi_check") as phi_span:
            scrubbed_summary, scrubbed_points, phi_count = _check_and_scrub_phi(
                summary_text, key_points
            )
            phi_span.set_attribute("phi_detected", phi_count > 0)
            phi_span.set_attribute("entities_scrubbed_count", phi_count)
            # Never record summary content in spans

        if phi_count > 0:
            logger.warning(
                "phi_detected_in_summarizer_output",
                extra={
                    "run_id": inp.run_id,
                    "entities_scrubbed_count": phi_count,
                    "agent_id": _AGENT_ID,
                    # Never log summary content
                },
            )
            phi_error = PipelineError(
                error_code=PHI_DETECTED,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=_AGENT_ID,
                run_id=inp.run_id,
                topic_id=topic_id,
                message="PHI detected in summarizer output; content scrubbed",
                retry_hint=(
                    "Retry summarizer; reinforce PHI avoidance in system prompt. "
                    "Scrubbing may have broken output coherence."
                ),
                context={
                    "entities_scrubbed_count": phi_count,
                    "retry_count": retry_count,
                },
            )
            record_span_error(parent_span, phi_error)
            log_pipeline_error(logger, phi_error)
            return SummarizerOutput(run_id=inp.run_id, success=False, error=phi_error)

        # ── Assemble output ───────────────────────────────────────────────────
        raw_conf = parsed.get("confidence", "MEDIUM")
        confidence: Literal["HIGH", "MEDIUM", "LOW"] = (
            raw_conf if raw_conf in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
        )

        citations = [
            Citation(
                index=c.get("index", idx + 1),
                url=c.get("url", ""),
                title=c.get("title", ""),
                quote=c.get("quote", ""),
            )
            for idx, c in enumerate(raw_citations)
            if c.get("url", "") in source_urls
        ]

        parent_span.set_attribute("summarizer.confidence", confidence)
        parent_span.set_attribute("summarizer.citation_count", len(citations))
        # Never record summary content in spans

    logger.info(
        "summarizer_complete",
        extra={
            "run_id": inp.run_id,
            "strategy": strategy,
            "source_count": len(sources),
            "citation_count": len(citations),
            "confidence": confidence,
            "retry_count": retry_count,
            # Never log summary content, topic text, or citations
        },
    )

    return SummarizerOutput(
        run_id=inp.run_id,
        success=True,
        summary=summary_text,
        key_points=key_points,
        citations=citations,
        confidence=confidence,
        topic_text=inp.topic_text or inp.article.title or "",
    )


@_ls_traceable(
    run_type="chain",
    name="summarizer",
    tags=["agent:summarizer"],
)
async def _traced_run_initial(inp: SummarizerInput, *, tracer: Any) -> SummarizerOutput:
    """LangSmith run name: 'summarizer' — for initial (non-retry) calls."""
    return await _core_run(inp, tracer=tracer)


@_ls_traceable(
    run_type="chain",
    name="summarizer-retry",
    tags=["agent:summarizer", "retry"],
)
async def _traced_run_retry(inp: SummarizerInput, *, tracer: Any) -> SummarizerOutput:
    """LangSmith run name: 'summarizer-retry' — for reviewer-feedback retry calls."""
    return await _core_run(inp, tracer=tracer)


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    inp: SummarizerInput,
    *,
    tracer: Any = None,
) -> SummarizerOutput:
    """
    Public entry point called by main.py and the Phase 2 orchestrator.

    MODEL_OVERRIDE is not applied — the summarizer always uses claude-sonnet-4-6.
    If inp.retry_count > 0, the LangSmith run is named 'summarizer-retry' so
    retry traces are distinguishable from initial calls in the UI.
    """
    load_dotenv()
    setup_telemetry("news-mas-summarizer")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    active_tracer = tracer if tracer is not None else get_tracer("summarizer")
    retry_count = inp.retry_count or 0

    # Tag the LangSmith run config (prompt version, run_id, tags)
    make_run_config("summarizer", _PROMPT_VERSION, run_id=inp.run_id)

    logger.info(
        "summarizer_start",
        extra={
            "run_id": inp.run_id,
            "source_count": len(inp.sources) if inp.sources else 1,
            "model_id": _MODEL_ID,
            "has_reviewer_feedback": bool(inp.reviewer_feedback),
            "retry_count": retry_count,
            "langsmith_active": langsmith_active,
            # Never log topic_text, article content, or reviewer_feedback
        },
    )

    # ── AAP token validation (never exempt in any environment) ────────────────
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
            return SummarizerOutput(run_id=inp.run_id, success=False, error=auth_error)

    # ── Route to correctly named LangSmith trace ──────────────────────────────
    if retry_count > 0:
        return await _traced_run_retry(inp, tracer=active_tracer)
    return await _traced_run_initial(inp, tracer=active_tracer)
