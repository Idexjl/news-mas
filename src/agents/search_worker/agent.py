"""
SearchWorker agent — discovers and returns news articles.

Pipeline for each topic in the request:
  1. Tavily search (advanced depth, include_raw_content=True)
  2. Content extraction, in priority order:
       a. Tavily raw_content  → confidence FULL,    content_source "tavily_extract"
       b. Jina Reader fallback → confidence PARTIAL, content_source "jina"
       c. Tavily snippet      → confidence SNIPPET,  content_source "snippet"
  3. PII/PHI scrub via Presidio → overrides confidence to SCRUBBED when entities found
  4. Prompt-injection check — discards result on match (confidence INJECTED)
  5. Assemble SearchResult

OTel spans emitted per topic:
  search_worker.search    (parent) → topic, run_id, result_count, articles_kept
  egress.tavily.search    (child)  → query_length, max_results, response_time_ms
  egress.http.fetch       (child)  → egress.host, content_source, content_length,
                                     pii_count, injection_detected
  search_worker.scrub     (child)  → confidence distribution + content_source counts
"""
from __future__ import annotations

import asyncio
import os
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv

try:
    from langsmith import traceable as _ls_traceable
except ImportError:  # graceful degradation if langsmith is unavailable
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec

from tavily import TavilyClient

from src.auth.token_service import validate_token
from src.common.observability import configure_langsmith, get_logger, get_meter, get_tracer, setup_telemetry
from src.common.pii_scrubber import scrub_text
from src.common.pipeline_errors import ResultConfidence
from src.common.prompt_loader import make_run_config
from src.common.schemas import SearchResult, SearchWorkerInput, SearchWorkerOutput
from src.common.security import ContentInjectionDetector

logger = get_logger(__name__)

_TOKEN_BUDGET: int = int(os.getenv("TOKEN_BUDGET_SOURCES", "200000"))
_REQUIRED_CAPABILITY = "search.web"

_INJECTION_THRESHOLD: float = float(os.getenv("INJECTION_DETECTION_THRESHOLD", "0.85"))
_content_detector = ContentInjectionDetector(threshold=_INJECTION_THRESHOLD)

# ── egress metrics ─────────────────────────────────────────────────────────────

_egress_instr: dict[str, Any] | None = None


def _get_egress_instruments() -> dict[str, Any]:
    global _egress_instr
    if _egress_instr is not None:
        return _egress_instr
    meter = get_meter("news-mas.search-worker")
    _egress_instr = {
        "tavily_calls": meter.create_counter(
            "mas.egress.tavily.calls",
            description="Total Tavily search API calls",
        ),
        "tavily_errors": meter.create_counter(
            "mas.egress.tavily.errors",
            description="Tavily search API errors",
        ),
        "tavily_query_length": meter.create_histogram(
            "mas.egress.tavily.query_length",
            description="Length of Tavily search query strings",
            unit="chars",
        ),
        "tavily_response_time": meter.create_histogram(
            "mas.egress.tavily.response_time_ms",
            description="Tavily search API response time",
            unit="ms",
        ),
        "fetch_calls": meter.create_counter(
            "mas.egress.fetch.calls",
            description="Total outbound content fetch calls",
        ),
        "injection_detected": meter.create_counter(
            "mas.egress.injection_detected",
            description="Prompt injection patterns detected in fetched content",
        ),
    }
    return _egress_instr


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def _validate_aap_token(token: str) -> dict[str, Any]:
    """Validate AAP token and assert search.web capability. Logs actor claims only."""
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


def _parse_published_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── Tavily call with days-parameter fallback ──────────────────────────────────

async def _tavily_search(
    client: TavilyClient,
    *,
    query: str,
    max_results: int,
    since_days: int,
) -> dict[str, Any]:
    """
    Call client.search() with recency filtering and raw content extraction.

    Uses search_depth from TAVILY_SEARCH_DEPTH env (default "advanced") and
    include_raw_content=True so Tavily returns pre-extracted article text,
    eliminating the need for a separate page fetcher.

    Falls back to a call without days= when the installed tavily-python version
    pre-dates that parameter (raises TypeError on an unexpected kwarg).
    """
    search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
    base: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_raw_content": True,
        "include_answer": False,
    }
    try:
        return await asyncio.to_thread(client.search, **base, days=since_days)
    except TypeError:
        logger.warning(
            "tavily_days_param_unsupported_retrying_without",
            extra={"query": query},
        )
        return await asyncio.to_thread(client.search, **base)


# ── Jina Reader fallback ──────────────────────────────────────────────────────

async def _fetch_with_jina(url: str) -> str | None:
    """
    Jina Reader converts any URL to clean markdown.
    Free, no API key, handles JS-rendered pages.

    [CONTENT-EXTRACTION-TODO] In production consider:
    - Jina Reader Pro for higher rate limits
    - Diffbot for enterprise content extraction
    - Browserless for JS-heavy sites
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://r.jina.ai/{url}",
                timeout=15.0,
                headers={"Accept": "text/markdown"},
            )
            if response.status_code == 200:
                content = response.text
                if len(content) > 200:
                    return content
        return None
    except Exception:
        return None


# ── per-result processing with egress.http.fetch span ─────────────────────────

async def _process_result(
    tracer: Any,
    result: dict[str, Any],
) -> SearchResult | None:
    """
    Extract, scrub, and injection-check one Tavily result.

    Content extraction priority:
      1. Tavily raw_content   → confidence FULL,    content_source "tavily_extract"
      2. Jina Reader fallback → confidence PARTIAL, content_source "jina"
      3. Tavily snippet       → confidence SNIPPET,  content_source "snippet"

    Injection detection and PHI scrub run on all content regardless of source.
    """
    url: str = result.get("url", "")
    title: str = result.get("title", "")
    snippet: str = result.get("content", "")
    hostname = urllib.parse.urlparse(url).hostname or url

    instr = _get_egress_instruments()

    # ── Content extraction ────────────────────────────────────────────────────
    # Confidence reflects content AVAILABILITY, not PHI status.
    # Threshold > 500 chars avoids treating a near-empty raw_content field as FULL.
    raw_content = result.get("raw_content")  # None when not present
    if raw_content and len(raw_content) > 500:
        raw_text = raw_content[:_TOKEN_BUDGET]
        base_confidence = ResultConfidence.FULL
        content_source = "tavily_extract"
    else:
        jina_text = await _fetch_with_jina(url)
        if jina_text:
            raw_text = jina_text[:_TOKEN_BUDGET]
            base_confidence = ResultConfidence.PARTIAL
            content_source = "jina"
        else:
            raw_text = snippet
            base_confidence = ResultConfidence.SNIPPET
            content_source = "snippet"

    with tracer.start_as_current_span("egress.http.fetch") as span:
        # Capture hostname only — full URL may contain sensitive query params.
        span.set_attribute("egress.host", hostname)
        span.set_attribute("egress.type", "content_fetch")
        span.set_attribute("egress.content_source", content_source)
        span.set_attribute("egress.content_length", len(raw_text))

        # PII/PHI scrub
        scrubbed_title = scrub_text(title)
        scrubbed_content = scrub_text(raw_text)

        pii_count = scrubbed_title.count("<") + scrubbed_content.count("<")
        span.set_attribute("egress.scrubbed", pii_count > 0)
        span.set_attribute("egress.pii_count", pii_count)

        # ML-based injection detection — DeBERTa understands context so security
        # articles, SQL tutorials, and sports reports are correctly identified as safe.
        is_safe_content, injection_confidence = _content_detector.is_safe(scrubbed_content)
        injection_detected = not is_safe_content

        span.set_attribute("egress.injection_detected", injection_detected)
        span.set_attribute("egress.injection_confidence", round(injection_confidence, 4))

        if injection_detected:
            logger.warning(
                "injection_detected_discarding_result",
                extra={
                    "egress_host": hostname,
                    "injection_confidence": round(injection_confidence, 4),
                    "detection_mode": "content_ml",
                    # Never log matched content.
                },
            )
            instr["injection_detected"].add(1)
            logger.warning(
                "egress.security.injection",
                extra={"egress_host": hostname, "injection_confidence": round(injection_confidence, 4)},
            )
            span.set_attribute("egress.confidence", ResultConfidence.INJECTED.value)
            # Return with empty content — injected payload is never forwarded.
            return SearchResult(
                url=url,
                title="",
                content="",
                source=result.get("source", ""),
                published_at=_parse_published_at(result.get("published_date")),
                confidence=ResultConfidence.INJECTED,
                confidence_reason=f"ml_injection:{injection_confidence:.2f}",
                content_source=content_source,
            )

        # Confidence reflects content availability (FULL/PARTIAL/SNIPPET), not PHI status.
        # PHI presence is recorded in confidence_reason for observability; it does not
        # downgrade confidence because the content is still fully extracted — just cleaned.
        confidence = base_confidence
        confidence_reason = (
            f"phi_entities:{pii_count}" if pii_count > 0
            else (None if base_confidence == ResultConfidence.FULL else base_confidence.value)
        )

        span.set_attribute("egress.confidence", confidence.value)
        instr["fetch_calls"].add(1)

    return SearchResult(
        url=url,
        title=scrubbed_title,
        content=scrubbed_content,
        published_at=_parse_published_at(result.get("published_date")),
        source=result.get("source", ""),
        confidence=confidence,
        confidence_reason=confidence_reason,
        content_source=content_source,
    )


# ── per-topic search ──────────────────────────────────────────────────────────

async def _search_topic(
    topic: str,
    inp: SearchWorkerInput,
    tracer: Any,
) -> list[SearchResult]:
    """Run Tavily search for one topic, extract content, scrub, return articles."""

    with tracer.start_as_current_span("search_worker.search") as span:
        span.set_attribute("topic", topic)
        span.set_attribute("run_id", inp.run_id)

        # ── egress.tavily.search span ─────────────────────────────────────────
        # [NETWORK-SEGMENTATION-TODO] In production (Azure Container Apps):
        # Outbound calls to api.tavily.com are controlled by egress policy on
        # the compute subnet. Only api.tavily.com and api.anthropic.com are in
        # the allowed egress list. Unexpected outbound hosts trigger a security
        # alert. This OTel span is the application-layer equivalent of that
        # network-level egress control.
        #
        # [SIEM-TODO] Egress anomalies on this span feed Sentinel network
        # analytics. The egress.tavily.search span attributes flow through
        # the OTel collector into Sentinel's network entity model:
        #   - tavily.query_length spikes → TavilyAnomalousQueryLength alert →
        #     Sentinel incident with "possible prompt injection in search path"
        #     description and automated Playbook to inspect the parent run.
        #   - mas.egress.tavily.calls_total surge → TavilyHighCallVolume alert
        #     → Sentinel incident + Playbook to pause the scheduler and notify
        #     the on-call operator.
        #   - egress.host appearing as anything other than "api.tavily.com" in
        #     an egress.http.fetch span → Sentinel network anomaly rule fires
        #     immediately, as the compute subnet should only reach Tavily and
        #     Anthropic endpoints (enforced at the NSG level; this is defence-
        #     in-depth at the application layer).
        #   Connect the dots in Sentinel by joining on run_id: the same run_id
        #   appears in the AAP token's aap_context, the egress spans, and the
        #   structured logs — giving a full audit trail from mint to egress.
        instr = _get_egress_instruments()
        tavily_start = time.monotonic()

        with tracer.start_as_current_span("egress.tavily.search") as tavily_span:
            tavily_span.set_attribute("egress.host", "api.tavily.com")
            tavily_span.set_attribute("egress.type", "search_api")
            tavily_span.set_attribute("egress.method", "search")
            tavily_span.set_attribute("tavily.query_length", len(topic))
            tavily_span.set_attribute("tavily.max_results", inp.max_results_per_topic)
            tavily_span.set_attribute("tavily.since_days", inp.since_days)
            tavily_span.set_attribute("run_id", inp.run_id)
            tavily_span.set_attribute("topic_id", topic)
            tavily_span.set_attribute("agent_id", "search-worker-01")

            # Anomaly: query too long (possible injection in search path)
            if len(topic) > 200:
                logger.warning(
                    "egress.anomaly.query_length",
                    extra={"query_length": len(topic)},
                )

            try:
                client = _get_tavily_client()
                response = await _tavily_search(
                    client,
                    query=topic,
                    max_results=inp.max_results_per_topic,
                    since_days=inp.since_days,
                )
            except Exception as exc:
                tavily_elapsed = (time.monotonic() - tavily_start) * 1000
                tavily_span.set_attribute("tavily.response_time_ms", tavily_elapsed)
                tavily_span.set_attribute("egress.error", True)
                tavily_span.set_attribute("egress.error_type", type(exc).__name__)
                if hasattr(exc, "status_code"):
                    tavily_span.set_attribute("egress.status_code", exc.status_code)
                instr["tavily_errors"].add(1)

                name = type(exc).__name__
                if "UsageLimit" in name or "quota" in str(exc).lower():
                    logger.warning(
                        "tavily_quota_exceeded",
                        extra={"topic": topic, "error": name},
                    )
                    return []
                if "Timeout" in name:
                    logger.warning("tavily_timeout", extra={"topic": topic})
                    return []
                raise

            tavily_elapsed = (time.monotonic() - tavily_start) * 1000
            results: list[dict[str, Any]] = response.get("results", [])

            tavily_span.set_attribute("tavily.results_returned", len(results))
            tavily_span.set_attribute("tavily.response_time_ms", tavily_elapsed)

            instr["tavily_calls"].add(1)
            instr["tavily_query_length"].record(len(topic))
            instr["tavily_response_time"].record(tavily_elapsed)

            logger.debug(
                "tavily_response_received",
                extra={
                    "topic": topic,
                    "response_keys": list(response.keys()),
                    "result_count": len(results),
                },
            )

        if not results:
            logger.warning("egress.anomaly.no_results", extra={"topic_id": topic})
            logger.info("tavily_no_results", extra={"topic": topic})
            span.set_attribute("result_count", 0)
            return []

        span.set_attribute("result_count", len(results))

        # Process all results concurrently — each creates its own Jina client if needed.
        processed: list[SearchResult | None | BaseException] = await asyncio.gather(
            *[_process_result(tracer, r) for r in results],
            return_exceptions=True,
        )

        articles: list[SearchResult] = []
        for item in processed:
            if isinstance(item, BaseException):
                logger.error(
                    "result_processing_failed",
                    extra={"error_type": type(item).__name__},
                )
            elif item is not None:
                articles.append(item)

        # Confidence distribution + content_source distribution span.
        with tracer.start_as_current_span("search_worker.scrub") as scrub_span:
            full_count = sum(1 for a in articles if a.confidence == ResultConfidence.FULL)
            snippet_count = sum(1 for a in articles if a.confidence == ResultConfidence.SNIPPET)
            injected_count = sum(1 for a in articles if a.confidence == ResultConfidence.INJECTED)
            tavily_extract_count = sum(1 for a in articles if a.content_source == "tavily_extract")
            jina_count = sum(1 for a in articles if a.content_source == "jina")
            snippet_source_count = sum(1 for a in articles if a.content_source == "snippet")
            scrub_span.set_attribute("confidence.full_count", full_count)
            scrub_span.set_attribute("confidence.snippet_count", snippet_count)
            scrub_span.set_attribute("confidence.injected_count", injected_count)
            scrub_span.set_attribute("search.tavily_extract_count", tavily_extract_count)
            scrub_span.set_attribute("search.jina_count", jina_count)
            scrub_span.set_attribute("search.snippet_count", snippet_source_count)

        span.set_attribute("articles_kept", len(articles))
        logger.info(
            "topic_search_complete",
            extra={
                "topic": topic,
                "results_raw": len(results),
                "articles_kept": len(articles),
                "run_id": inp.run_id,
            },
        )

    return articles


# ── LangSmith-traced entry point ──────────────────────────────────────────────

@_ls_traceable(
    run_type="retriever",
    name="search_worker",
    tags=["agent:search_worker"],
)
async def _traced_run(inp: SearchWorkerInput, *, tracer: Any = None) -> SearchWorkerOutput:
    """
    LangSmith-traced body. ``inp`` is captured as the run input so run_id,
    topics, and since_days appear in every LangSmith trace automatically.
    ``tracer`` is injected by tests; production uses the global OTel tracer.
    """
    active_tracer = tracer if tracer is not None else get_tracer("search-worker")

    topic_results: list[list[SearchResult] | BaseException] = await asyncio.gather(
        *[_search_topic(topic, inp, active_tracer) for topic in inp.topics],
        return_exceptions=True,
    )

    articles: list[SearchResult] = []
    connection_errors: list[str] = []
    for topic, result in zip(inp.topics, topic_results):
        if isinstance(result, BaseException):
            error_type = type(result).__name__
            logger.error(
                "topic_search_failed",
                extra={
                    "topic": topic,
                    "error_type": error_type,
                    "run_id": inp.run_id,
                },
            )
            if isinstance(result, (ConnectionError, OSError)):
                connection_errors.append(error_type)
        else:
            articles.extend(result)

    warnings: list[str] = []
    if connection_errors and not articles:
        warnings.append(f"tavily_connection_error:{','.join(connection_errors)}")

    # Rough token estimate (4 chars ≈ 1 token) for LangSmith metadata.
    token_estimate = sum(len(a.content) for a in articles) // 4
    logger.info(
        "search_worker_complete",
        extra={
            "run_id": inp.run_id,
            "article_count": len(articles),
            "token_estimate": token_estimate,
        },
    )

    return SearchWorkerOutput(run_id=inp.run_id, articles=articles, warnings=warnings)


# ── public entry point ────────────────────────────────────────────────────────

async def run_agent(inp: SearchWorkerInput, *, tracer: Any = None) -> SearchWorkerOutput:
    """
    Public entry point. ``tracer`` is optional; pass a test-scoped tracer to
    capture spans without touching the global OTel provider.
    """
    load_dotenv()
    setup_telemetry("news-mas-search-worker")
    langsmith_active = configure_langsmith()
    logger.info("agent.startup", extra={
        "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "langsmith_enabled": langsmith_active,
        "service_name": os.getenv("SERVICE_NAME"),
    })

    run_cfg = make_run_config("search_worker", run_id=inp.run_id)
    logger.info(
        "search_worker_start",
        extra={
            "run_id": inp.run_id,
            "topic_count": len(inp.topics),
            "since_days": inp.since_days,
            "langsmith_run_name": run_cfg["run_name"],
        },
    )

    if inp.aap_token:
        _validate_aap_token(inp.aap_token)

    return await _traced_run(inp, tracer=tracer)
