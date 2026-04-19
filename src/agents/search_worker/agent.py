"""
SearchWorker agent — discovers, fetches, scrubs, and returns news articles.

Pipeline for each topic in the request:
  1. Tavily search  → article URLs + snippets
  2. Concurrent httpx page fetch (async, 15 s timeout per URL)
  3. BeautifulSoup HTML strip → plain text
  4. Truncate to TOKEN_BUDGET_SOURCES chars
  5. PII/PHI scrub via Presidio
  6. Prompt-injection check — discard result on match
  7. Assemble RawArticle

OTel spans emitted per topic:
  search_worker.search  (parent)  → topic, run_id, query, result_count
  search_worker.fetch   (child)   → url, content_length
  search_worker.scrub   (child)   → url, scrubbed_count
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

try:
    from langsmith import traceable as _ls_traceable
except ImportError:  # graceful degradation if langsmith is unavailable
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec

from tavily import TavilyClient

from src.auth.token_service import validate_token
from src.common.observability import get_logger, get_tracer
from src.common.pii_scrubber import scrub_text
from src.common.prompt_loader import make_run_config
from src.common.schemas import RawArticle, SearchWorkerInput, SearchWorkerOutput
from src.common.security import detect_injection

logger = get_logger(__name__)

_FETCH_TIMEOUT: float = 15.0
_TOKEN_BUDGET: int = int(os.getenv("TOKEN_BUDGET_SOURCES", "200000"))
_REQUIRED_CAPABILITY = "search.web"

# Tags stripped from fetched HTML before extracting text.
_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript"]


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


async def _fetch_page(http: httpx.AsyncClient, url: str) -> str:
    """Fetch *url* and return stripped plain text. Raises on HTTP/timeout error."""
    resp = await http.get(url, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


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
    Call client.search() with recency filtering.

    Falls back to a call without days= when the installed tavily-python version
    pre-dates that parameter (raises TypeError on an unexpected kwarg).
    """
    base: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    try:
        return await asyncio.to_thread(client.search, **base, days=since_days)
    except TypeError:
        logger.warning(
            "tavily_days_param_unsupported_retrying_without",
            extra={"query": query},
        )
        return await asyncio.to_thread(client.search, **base)


# ── per-topic search ──────────────────────────────────────────────────────────

async def _search_topic(
    topic: str,
    inp: SearchWorkerInput,
    tracer: Any,
) -> list[RawArticle]:
    """Run Tavily search for one topic, fetch pages concurrently, scrub, return articles."""

    with tracer.start_as_current_span("search_worker.search") as span:
        span.set_attribute("topic", topic)
        span.set_attribute("run_id", inp.run_id)
        span.set_attribute("query", topic)

        # Tavily is synchronous — run in thread to avoid blocking the event loop.
        try:
            client = _get_tavily_client()
            response = await _tavily_search(
                client,
                query=topic,
                max_results=inp.max_results_per_topic,
                since_days=inp.since_days,
            )
        except Exception as exc:
            name = type(exc).__name__
            if "UsageLimit" in name or "quota" in str(exc).lower():
                logger.warning("tavily_quota_exceeded", extra={"topic": topic, "error": name})
                return []
            if "Timeout" in name:
                logger.warning("tavily_timeout", extra={"topic": topic})
                return []
            raise

        logger.debug(
            "tavily_response_received",
            extra={
                "topic": topic,
                "response_keys": list(response.keys()),
                "result_count": len(response.get("results", [])),
            },
        )

        results: list[dict[str, Any]] = response.get("results", [])
        if not results:
            logger.info("tavily_no_results", extra={"topic": topic})
            span.set_attribute("result_count", 0)
            return []

        span.set_attribute("result_count", len(results))

        # Fetch all pages concurrently; failures fall back to the Tavily snippet.
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as http:
            pages: list[str | BaseException] = await asyncio.gather(
                *[_fetch_page(http, r["url"]) for r in results],
                return_exceptions=True,
            )

        articles: list[RawArticle] = []

        for result, page in zip(results, pages):
            url: str = result.get("url", "")
            title: str = result.get("title", "")
            snippet: str = result.get("content", "")

            # ── fetch span ────────────────────────────────────────────────────
            with tracer.start_as_current_span("search_worker.fetch") as fetch_span:
                fetch_span.set_attribute("url", url)
                if isinstance(page, BaseException):
                    logger.info(
                        "page_fetch_fallback",
                        extra={"url": url, "error_type": type(page).__name__},
                    )
                    raw_text = snippet
                else:
                    raw_text = page[:_TOKEN_BUDGET]
                fetch_span.set_attribute("content_length", len(raw_text))

            # ── scrub + injection check span ──────────────────────────────────
            with tracer.start_as_current_span("search_worker.scrub") as scrub_span:
                scrub_span.set_attribute("url", url)

                scrubbed_title = scrub_text(title)
                scrubbed_content = scrub_text(raw_text)
                scrubbed_snippet = scrub_text(snippet)

                # Count Presidio placeholders as a proxy for entities scrubbed.
                # Post-BS4 text contains no raw HTML, so '<' means a placeholder.
                scrubbed_count = (
                    scrubbed_title.count("<")
                    + scrubbed_content.count("<")
                    + scrubbed_snippet.count("<")
                )
                scrub_span.set_attribute("scrubbed_count", scrubbed_count)

                # Injection check runs on the already-scrubbed content.
                patterns = detect_injection(scrubbed_content)
                if patterns:
                    logger.warning(
                        "injection_detected_discarding_result",
                        extra={
                            "url": url,
                            "pattern_count": len(patterns),
                            # Never log matched patterns or raw content.
                        },
                    )
                    continue

                logger.debug(
                    "result_passed_injection_check",
                    extra={"url": url},
                )

            articles.append(
                RawArticle(
                    url=url,
                    title=scrubbed_title,
                    content=scrubbed_content,
                    published_at=_parse_published_at(result.get("published_date")),
                    source=result.get("source", ""),
                )
            )

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

    topic_results: list[list[RawArticle] | BaseException] = await asyncio.gather(
        *[_search_topic(topic, inp, active_tracer) for topic in inp.topics],
        return_exceptions=True,
    )

    articles: list[RawArticle] = []
    for topic, result in zip(inp.topics, topic_results):
        if isinstance(result, BaseException):
            logger.error(
                "topic_search_failed",
                extra={
                    "topic": topic,
                    "error_type": type(result).__name__,
                    "run_id": inp.run_id,
                },
            )
        else:
            articles.extend(result)

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

    return SearchWorkerOutput(run_id=inp.run_id, articles=articles)


# ── public entry point ────────────────────────────────────────────────────────

async def run_agent(inp: SearchWorkerInput, *, tracer: Any = None) -> SearchWorkerOutput:
    """
    Public entry point. ``tracer`` is optional; pass a test-scoped tracer to
    capture spans without touching the global OTel provider.
    """
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
