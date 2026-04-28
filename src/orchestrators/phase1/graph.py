"""
Phase 1 LangGraph orchestrator.

Pipeline: SearchWorker → HeatScorer → FilterAgent → Selector → Phase1Judge

Fan-out pattern
--------------
Each per-topic stage is fanned out via LangGraph's Send() API so topics are
processed in parallel.  Routing functions (fan_out_search, fan_out_heat_score,
fan_out_filter) act as conditional edges from their respective waypoint nodes and
return a list of Send() objects — one per viable topic.  After all parallel nodes
in a batch complete, LangGraph merges their state updates via the Annotated field
reducers and resumes execution from the next node.

Error routing
-------------
  FATAL agent result  → skip that topic, add tombstone, continue
  RETRYABLE result    → retry once inline, then treat as FATAL
  DEGRADED result     → continue with flagged warnings
  No candidates survive selector → run_status = "no_candidates"
  Selector FATAL      → abort, run_status = "failed"
  Judge FATAL         → DEGRADED fallback (selector choices unchanged)

Circuit breaker
---------------
Checks RunRepository for any "running" run at startup.  If one exists, the new
run is aborted immediately with run_status = "circuit_breaker_active".
Gracefully skipped when FERNET_KEY is not set (local dev without storage).

Token budget
------------
Tracks cumulative token estimates across all agent calls.  When total exceeds
MAX_TOKENS_PER_RUN the run is aborted with run_status = "budget_exceeded".

[DPOP-TODO] All agent calls should use DPoP-bound Entra ID tokens once the
auth layer is active.  See phase2/graph.py [DPOP-TODO] block for the template.
"""
from __future__ import annotations

import operator
import os
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.types import Send
from langgraph.graph import END, StateGraph

from src.auth.token_service import exchange_token, mint_token
from src.common.error_codes import (
    BUDGET_EXHAUSTED,
    CIRCUIT_BREAKER_ACTIVE,
    NO_VIABLE_CANDIDATES,
)
from src.common.observability import get_logger, get_tracer, setup_telemetry
from src.common.pipeline_errors import AgentResult, ErrorSeverity, PipelineError
from src.common.schemas import (
    CandidateConfidence,
    FilterAgentInput,
    FilterAgentOutput,
    HeatScorerInput,
    HeatScorerOutput,
    JudgedArticle,
    Phase1JudgeInput,
    Phase1JudgeOutput,
    Phase1Output,
    RankedCandidate,
    SearchWorkerInput,
    SearchWorkerOutput,
    ScoredFilteredTopic,
    SelectorInput,
    SelectorOutput,
    TopicInput,
)

load_dotenv()
setup_telemetry("news-mas-orchestrator-1")
logger = get_logger(__name__)

_MAX_TOKENS_DEFAULT = 2_000_000
_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


# ── State reducers ─────────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


# ── Phase1State ────────────────────────────────────────────────────────────────

class Phase1State(TypedDict):
    run_id: str
    since_date: str
    user_id: str
    topics: list[TopicInput]
    # Annotated fields use reducers so parallel Send() nodes can update them
    search_results: Annotated[dict[str, SearchWorkerOutput], _merge_dicts]
    heat_scores: Annotated[dict[str, HeatScorerOutput], _merge_dicts]
    filtered_results: Annotated[dict[str, FilterAgentOutput], _merge_dicts]
    scored_filtered_topics: list[ScoredFilteredTopic]
    selector_output: Optional[SelectorOutput]
    judge_output: Optional[Phase1JudgeOutput]
    ranked_candidates: list[RankedCandidate]
    errors: Annotated[list[PipelineError], operator.add]
    run_status: str
    aap_token: Optional[str]
    total_tokens_used: Annotated[int, operator.add]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _since_date_to_days(since_date: str) -> int:
    """Convert an ISO date string to an integer day count for SearchWorkerInput."""
    try:
        dt = datetime.fromisoformat(since_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).days
        return max(1, min(90, days))
    except (ValueError, TypeError):
        return int(os.getenv("SINCE_DATE_DEFAULT_DAYS", "7"))


def _max_tokens() -> int:
    return int(os.getenv("MAX_TOKENS_PER_RUN", str(_MAX_TOKENS_DEFAULT)))


def _secret() -> str:
    return os.getenv("MAS_SECRET_KEY", "")


def _try_get_run_repo():
    """Return RunRepository if FERNET_KEY is set, else None (local dev fallback)."""
    try:
        from src.common.storage import FernetStorage, RunRepository
        return RunRepository(FernetStorage())
    except (RuntimeError, Exception):
        return None


def _mint_orchestrator_token(run_id: str, user_id: str) -> Optional[str]:
    """Mint a run-scoped AAP token for the orchestrator. Returns None if MAS_SECRET_KEY unset."""
    secret = _secret()
    if not secret:
        return None
    try:
        return mint_token(
            subject=f"orchestrator:{user_id}",
            agent_name="phase1-orchestrator",
            model_provider="none",
            model_id="none",
            capabilities=["orchestrate.phase1"],
            task_id=run_id,
            task_description="Phase 1 discovery and scoring pipeline",
            data_sensitivity="standard",
            oversight_required=False,
            oversight_level="none",
            context={"run_id": run_id, "user_id": user_id},
            secret_key=secret,
        )
    except Exception as exc:
        logger.warning("aap_token_mint_failed", extra={"error_type": type(exc).__name__})
        return None


def _exchange_for_agent(token: Optional[str], agent_name: str, capabilities: list[str]) -> Optional[str]:
    """Exchange the orchestrator token for an agent-scoped token."""
    if not token:
        return None
    secret = _secret()
    if not secret:
        return None
    # model_provider/model_id are metadata only — not validated here
    _MODEL_PROVIDERS = {
        "search-worker": ("tavily", "none"),
        "heat-scorer": ("ollama", "gemma4:e4b"),
        "filter-agent": ("ollama", "gemma4:e4b"),
        "selector": ("ollama", "gemma4:e4b"),
        "phase1-judge": ("ollama", "gemma4:e4b"),
    }
    provider, model_id = _MODEL_PROVIDERS.get(agent_name, ("none", "none"))
    try:
        return exchange_token(
            token=token,
            new_agent_name=agent_name,
            new_model_provider=provider,
            new_model_id=model_id,
            new_capabilities=capabilities,
            secret_key=secret,
        )
    except Exception as exc:
        logger.warning(
            "aap_token_exchange_failed",
            extra={"agent": agent_name, "error_type": type(exc).__name__},
        )
        return None


def _estimate_tokens(text: str) -> int:
    """Rough 4-chars-per-token estimate."""
    return max(0, len(text) // 4)


def _make_tombstone_error(
    topic: TopicInput,
    run_id: str,
    reason: str,
    agent_id: str = "phase1-orchestrator",
) -> PipelineError:
    return PipelineError(
        error_code=NO_VIABLE_CANDIDATES,
        severity=ErrorSeverity.SKIP,
        agent_id=agent_id,
        run_id=run_id,
        topic_id=topic.topic_id,
        message=reason,
    )


# ── Node: initialize_run ───────────────────────────────────────────────────────

async def initialize_run(state: Phase1State) -> dict:
    """
    Set up run state, check circuit breaker, mint AAP token, persist initial
    RunState.  Sets run_status = "circuit_breaker_active" or "budget_exceeded"
    to trigger early-exit routing before any agent calls.
    """
    run_id = state["run_id"]
    tracer = get_tracer("news-mas-orchestrator-1")

    with tracer.start_as_current_span("phase1.initialize") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("topic_count", len(state["topics"]))
        span.set_attribute("since_date", state["since_date"])

        if not state["topics"]:
            logger.warning("phase1_no_topics", extra={"run_id": run_id})
            return {"run_status": "no_topics", "errors": [
                PipelineError(
                    error_code=NO_VIABLE_CANDIDATES,
                    severity=ErrorSeverity.FATAL,
                    agent_id="phase1-orchestrator",
                    run_id=run_id,
                    message="No topics provided to Phase 1 pipeline",
                )
            ]}

        # ── Circuit breaker ─────────────────────────────────────────────────
        _cb_stale_minutes = int(os.environ.get("CIRCUIT_BREAKER_STALE_MINUTES", "90"))
        repo = _try_get_run_repo()
        if repo is not None:
            try:
                run_ids = repo.list_runs()
                for rid in run_ids:
                    if rid == run_id:
                        continue
                    try:
                        existing = repo.load(rid)
                        if existing.get("status") != "running":
                            continue
                        # Auto-expire runs that pre-date the stale threshold — they
                        # crashed without writing a final status and should not block.
                        started_at_str = existing.get("started_at")
                        if started_at_str:
                            try:
                                started = datetime.fromisoformat(started_at_str)
                                if started.tzinfo is None:
                                    started = started.replace(tzinfo=timezone.utc)
                                age_min = (datetime.now(timezone.utc) - started).total_seconds() / 60
                                if age_min > _cb_stale_minutes:
                                    logger.warning(
                                        "circuit_breaker_stale_run_expired",
                                        extra={"run_id": run_id, "stale_run_id": rid, "age_minutes": round(age_min, 1)},
                                    )
                                    try:
                                        repo.save(rid, {**existing, "status": "crashed"})
                                    except Exception:
                                        pass
                                    continue
                            except (ValueError, TypeError):
                                pass
                        logger.warning(
                            "circuit_breaker_triggered",
                            extra={"run_id": run_id, "blocking_run": rid},
                        )
                        span.set_attribute("circuit_breaker_triggered", True)
                        span.set_attribute("blocking_run_id", rid)
                        return {
                            "run_status": "circuit_breaker_active",
                            "errors": [PipelineError(
                                error_code=CIRCUIT_BREAKER_ACTIVE,
                                severity=ErrorSeverity.SKIP,
                                agent_id="phase1-orchestrator",
                                run_id=run_id,
                                message="Another run is already in progress",
                                context={"blocking_run_id": rid},
                            )],
                        }
                    except (KeyError, Exception):
                        continue
            except Exception as exc:
                logger.warning(
                    "circuit_breaker_check_failed",
                    extra={"run_id": run_id, "error_type": type(exc).__name__},
                )

            # ── Persist initial run state ───────────────────────────────────
            try:
                repo.save(run_id, {
                    "run_id": run_id,
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "topics": [t.topic_id for t in state["topics"]],
                })
            except Exception as exc:
                logger.warning(
                    "run_state_save_failed",
                    extra={"run_id": run_id, "error_type": type(exc).__name__},
                )

        # ── Mint run-scoped AAP token ───────────────────────────────────────
        aap_token = _mint_orchestrator_token(run_id, state["user_id"])

        logger.info(
            "phase1_initialized",
            extra={
                "run_id": run_id,
                "topic_count": len(state["topics"]),
                "aap_token_minted": aap_token is not None,
            },
        )

    return {"run_status": "running", "aap_token": aap_token}


# ── Fan-out edge: search ───────────────────────────────────────────────────────

def fan_out_search(state: Phase1State):
    """
    Conditional edge from initialize_run.
    Returns one Send("search_topic", payload) per topic, dispatching parallel searches.
    Returns the string "finalize_phase1" on early-exit conditions so LangGraph passes
    the full accumulated state (not an empty dict) to finalize_phase1.
    """
    if state["run_status"] in ("circuit_breaker_active", "no_topics"):
        return "finalize_phase1"

    return [
        Send("search_topic", {
            "topic": topic,
            "run_id": state["run_id"],
            "since_date": state["since_date"],
            "aap_token": state["aap_token"],
        })
        for topic in state["topics"]
    ]


# ── Node: search_topic (parallel, one per topic) ───────────────────────────────

async def search_topic(state: dict) -> dict:
    """
    Search one topic.  Runs in parallel with other search_topic instances.
    Merges results into Phase1State.search_results via _merge_dicts reducer.
    """
    from src.orchestrators.phase1.registry_client import call_search_worker

    topic: TopicInput = state["topic"]
    run_id: str = state["run_id"]
    since_date: str = state["since_date"]
    aap_token: Optional[str] = state.get("aap_token")

    since_days = _since_date_to_days(since_date)
    agent_token = _exchange_for_agent(aap_token, "search-worker", ["search.web"])

    tracer = get_tracer("news-mas-orchestrator-1")
    with tracer.start_as_current_span("phase1.topic.search") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("topic_id", topic.topic_id)
        span.set_attribute("since_days", since_days)

        try:
            inp = SearchWorkerInput(
                run_id=run_id,
                topics=[topic.topic_text],
                since_days=since_days,
                aap_token=agent_token,
            )
            output = await call_search_worker(inp)

            token_est = sum(_estimate_tokens(a.content) for a in output.articles)
            span.set_attribute("article_count", len(output.articles))
            span.set_attribute("token_estimate", token_est)

            if not output.success:
                err = output.error or PipelineError(
                    error_code="SEARCH_FAILED",
                    severity=ErrorSeverity.SKIP,
                    agent_id="search-worker",
                    run_id=run_id,
                    topic_id=topic.topic_id,
                    message="Search worker returned failure",
                )
                logger.warning(
                    "search_topic_failed",
                    extra={"run_id": run_id, "topic_id": topic.topic_id,
                           "error_code": err.error_code},
                )
                return {
                    "search_results": {topic.topic_id: output},
                    "errors": [err],
                    "total_tokens_used": token_est,
                }

            return {
                "search_results": {topic.topic_id: output},
                "errors": [],
                "total_tokens_used": token_est,
            }

        except Exception as exc:
            err = PipelineError(
                error_code="SEARCH_EXCEPTION",
                severity=ErrorSeverity.SKIP,
                agent_id="search-worker",
                run_id=run_id,
                topic_id=topic.topic_id,
                message="Unexpected exception in search_topic node",
                context={"error_type": type(exc).__name__},
            )
            logger.error(
                "search_topic_exception",
                extra={"run_id": run_id, "topic_id": topic.topic_id,
                       "error_type": type(exc).__name__},
            )
            span.set_attribute("exception", True)
            failed_output = SearchWorkerOutput(
                run_id=run_id, success=False, error=err, articles=[]
            )
            return {
                "search_results": {topic.topic_id: failed_output},
                "errors": [err],
                "total_tokens_used": 0,
            }


# ── Node: fan_out_heat_score (waypoint between search and heat-score stages) ───

async def fan_out_heat_score(state: Phase1State) -> dict:
    """
    Waypoint node.  Routing is handled by _route_heat_score conditional edge.
    Checks the token budget here so the result can be written back to graph state
    before the routing function reads it.
    """
    if state.get("total_tokens_used", 0) > _max_tokens():
        logger.warning(
            "budget_exceeded_before_heat_score",
            extra={"run_id": state["run_id"],
                   "total_tokens_used": state.get("total_tokens_used", 0)},
        )
        return {"run_status": "budget_exceeded"}
    return {}


def _route_heat_score(state: Phase1State):
    """
    Conditional edge from fan_out_heat_score.
    Fans out to heat_score_topic for topics with successful search results.
    Returns strings (not Send) for non-fan-out paths so LangGraph passes the full
    accumulated state to the target node instead of an empty dict.
    """
    if state.get("run_status") == "budget_exceeded":
        return "finalize_phase1"

    topic_map = {t.topic_id: t for t in state["topics"]}
    sends = []
    for topic_id, search_out in state.get("search_results", {}).items():
        if search_out.success and search_out.articles:
            sends.append(Send("heat_score_topic", {
                "topic_id": topic_id,
                "topic": topic_map.get(topic_id),
                "articles": search_out.articles,
                "run_id": state["run_id"],
                "aap_token": state["aap_token"],
            }))

    if not sends:
        logger.info(
            "no_viable_topics_for_heat_score",
            extra={"run_id": state["run_id"]},
        )
        return "collect_phase1_results"

    return sends


# ── Node: heat_score_topic (parallel, one per topic) ──────────────────────────

async def heat_score_topic(state: dict) -> dict:
    """
    Heat-score one topic's articles.  Runs in parallel with other heat_score_topic
    instances.  Merges results into Phase1State.heat_scores via _merge_dicts reducer.
    """
    from src.orchestrators.phase1.registry_client import call_heat_scorer

    topic_id: str = state["topic_id"]
    topic: Optional[TopicInput] = state.get("topic")
    articles = state["articles"]
    run_id: str = state["run_id"]
    aap_token: Optional[str] = state.get("aap_token")

    agent_token = _exchange_for_agent(aap_token, "heat-scorer", ["score.heat"])

    tracer = get_tracer("news-mas-orchestrator-1")
    with tracer.start_as_current_span("phase1.topic.heat_score") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("topic_id", topic_id)
        span.set_attribute("article_count", len(articles))

        try:
            inp = HeatScorerInput(
                run_id=run_id,
                articles=articles,
                aap_token=agent_token,
            )
            output = await call_heat_scorer(inp)
            token_est = len(articles) * 200  # ~200 tokens per article (snippet only)

            span.set_attribute("scored_count", len(output.scored_articles))
            span.set_attribute(
                "overall_confidence",
                output.candidate_confidence.overall_confidence
                if output.candidate_confidence else "UNKNOWN",
            )

            if not output.success:
                err = output.error or PipelineError(
                    error_code="HEAT_SCORE_FAILED",
                    severity=ErrorSeverity.SKIP,
                    agent_id="heat-scorer",
                    run_id=run_id,
                    topic_id=topic_id,
                    message="Heat scorer returned failure",
                )
                return {
                    "heat_scores": {topic_id: output},
                    "errors": [err],
                    "total_tokens_used": token_est,
                }

            return {
                "heat_scores": {topic_id: output},
                "errors": [],
                "total_tokens_used": token_est,
            }

        except Exception as exc:
            err = PipelineError(
                error_code="HEAT_SCORE_EXCEPTION",
                severity=ErrorSeverity.SKIP,
                agent_id="heat-scorer",
                run_id=run_id,
                topic_id=topic_id,
                message="Unexpected exception in heat_score_topic node",
                context={"error_type": type(exc).__name__},
            )
            logger.error(
                "heat_score_topic_exception",
                extra={"run_id": run_id, "topic_id": topic_id,
                       "error_type": type(exc).__name__},
            )
            failed = HeatScorerOutput(
                run_id=run_id, success=False, error=err, scored_articles=[]
            )
            return {
                "heat_scores": {topic_id: failed},
                "errors": [err],
                "total_tokens_used": 0,
            }


# ── Node: fan_out_filter (waypoint between heat-score and filter stages) ───────

async def fan_out_filter(state: Phase1State) -> dict:
    """
    Waypoint node.  Routing is handled by _route_filter conditional edge.
    Checks the token budget here so the result can be written back to graph state
    before the routing function reads it.
    """
    if state.get("total_tokens_used", 0) > _max_tokens():
        logger.warning(
            "budget_exceeded_before_filter",
            extra={"run_id": state["run_id"],
                   "total_tokens_used": state.get("total_tokens_used", 0)},
        )
        return {"run_status": "budget_exceeded"}
    return {}


def _route_filter(state: Phase1State):
    """
    Conditional edge from fan_out_filter.
    Fans out to filter_topic for topics with successful heat scores.
    Returns strings (not Send) for non-fan-out paths so LangGraph passes the full
    accumulated state to the target node instead of an empty dict.
    """
    if state.get("run_status") == "budget_exceeded":
        return "finalize_phase1"

    topic_map = {t.topic_id: t for t in state["topics"]}
    search_results = state.get("search_results", {})
    sends = []

    for topic_id, heat_out in state.get("heat_scores", {}).items():
        if heat_out.success and heat_out.scored_articles:
            topic = topic_map.get(topic_id)
            search_out = search_results.get(topic_id)
            sr_articles = search_out.articles if (search_out and search_out.success) else []
            sends.append(Send("filter_topic", {
                "topic_id": topic_id,
                "topic": topic,
                "scored_articles": heat_out.scored_articles,
                "search_results_for_topic": sr_articles,
                "run_id": state["run_id"],
                "aap_token": state["aap_token"],
            }))

    if not sends:
        logger.info(
            "no_viable_topics_for_filter",
            extra={"run_id": state["run_id"]},
        )
        return "collect_phase1_results"

    return sends


# ── Node: filter_topic (parallel, one per topic) ──────────────────────────────

async def filter_topic(state: dict) -> dict:
    """
    Semantically filter one topic's articles.  Runs in parallel.
    Merges results into Phase1State.filtered_results via _merge_dicts reducer.
    """
    from src.orchestrators.phase1.registry_client import call_filter_agent

    topic_id: str = state["topic_id"]
    topic: Optional[TopicInput] = state.get("topic")
    scored_articles = state["scored_articles"]
    search_results_for_topic = state.get("search_results_for_topic", [])
    run_id: str = state["run_id"]
    aap_token: Optional[str] = state.get("aap_token")

    constraints = topic.constraints if topic else []
    agent_token = _exchange_for_agent(aap_token, "filter-agent", ["filter.content"])

    tracer = get_tracer("news-mas-orchestrator-1")
    with tracer.start_as_current_span("phase1.topic.filter") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("topic_id", topic_id)
        span.set_attribute("input_count", len(scored_articles))
        span.set_attribute("constraints_count", len(constraints))

        try:
            inp = FilterAgentInput(
                run_id=run_id,
                scored_articles=scored_articles,
                constraints=constraints,
                search_results=search_results_for_topic,
                aap_token=agent_token,
            )
            output = await call_filter_agent(inp)
            token_est = len(scored_articles) * 150  # snippet-only estimate

            span.set_attribute("kept_count", len(output.kept_results))
            span.set_attribute("dropped_count", output.dropped_count)

            if not output.success and output.error:
                sev = output.error.severity
                if sev == ErrorSeverity.FATAL:
                    return {
                        "filtered_results": {topic_id: output},
                        "errors": [output.error],
                        "total_tokens_used": token_est,
                    }
                # SKIP / RETRYABLE — treat as no-keep (tombstone downstream)
                return {
                    "filtered_results": {topic_id: output},
                    "errors": [output.error],
                    "total_tokens_used": token_est,
                }

            return {
                "filtered_results": {topic_id: output},
                "errors": [],
                "total_tokens_used": token_est,
            }

        except Exception as exc:
            err = PipelineError(
                error_code="FILTER_EXCEPTION",
                severity=ErrorSeverity.SKIP,
                agent_id="filter-agent",
                run_id=run_id,
                topic_id=topic_id,
                message="Unexpected exception in filter_topic node",
                context={"error_type": type(exc).__name__},
            )
            logger.error(
                "filter_topic_exception",
                extra={"run_id": run_id, "topic_id": topic_id,
                       "error_type": type(exc).__name__},
            )
            failed = FilterAgentOutput(
                run_id=run_id, success=False, error=err,
                kept_results=[], removed_results=[], removal_reasons=[],
                filtered_articles=[], dropped_count=0,
            )
            return {
                "filtered_results": {topic_id: failed},
                "errors": [err],
                "total_tokens_used": 0,
            }


# ── Node: collect_phase1_results ───────────────────────────────────────────────

async def collect_phase1_results(state: Phase1State) -> dict:
    """
    Collect per-topic results from all three parallel stages and assemble the
    ScoredFilteredTopic list that the Selector needs.  Tombstone topics that
    failed at any stage.
    """
    run_id = state["run_id"]
    search_results = state.get("search_results", {})
    heat_scores = state.get("heat_scores", {})
    filtered_results = state.get("filtered_results", {})

    scored_filtered: list[ScoredFilteredTopic] = []
    new_errors: list[PipelineError] = []

    for topic in state["topics"]:
        tid = topic.topic_id

        search_out: Optional[SearchWorkerOutput] = search_results.get(tid)
        heat_out: Optional[HeatScorerOutput] = heat_scores.get(tid)
        filter_out: Optional[FilterAgentOutput] = filtered_results.get(tid)

        # Topic tombstoned if it never reached filter (search or heat failed)
        if not filter_out or not filter_out.success or not filter_out.kept_results:
            if filter_out and filter_out.error:
                reason = f"{filter_out.error.error_code}: {filter_out.error.message}"
            elif heat_out and heat_out.error:
                reason = f"{heat_out.error.error_code}: {heat_out.error.message}"
            elif search_out and search_out.error:
                reason = f"{search_out.error.error_code}: {search_out.error.message}"
            else:
                reason = "No articles survived scoring and filtering"
            new_errors.append(_make_tombstone_error(topic, run_id, reason))
            continue

        # Compute representative heat score and confidence
        avg_heat = (
            sum(sa.heat_score for sa in filter_out.kept_results) / len(filter_out.kept_results)
            if filter_out.kept_results else 0.0
        )
        confidence_obj: Optional[CandidateConfidence] = (
            heat_out.candidate_confidence if heat_out else None
        )
        overall_conf = (
            confidence_obj.overall_confidence if confidence_obj else "LOW"
        )

        scored_filtered.append(ScoredFilteredTopic(
            topic_id=tid,
            topic_text=topic.topic_text,
            heat_score=avg_heat,
            kept_results=filter_out.kept_results,
            confidence=overall_conf,
            removal_reasons=filter_out.removal_reasons,
        ))

    logger.info(
        "phase1_results_collected",
        extra={
            "run_id": run_id,
            "viable_topic_count": len(scored_filtered),
            "tombstoned_count": len(new_errors),
        },
    )

    return {
        "scored_filtered_topics": scored_filtered,
        "errors": new_errors,
    }


# ── Edge: after collect ────────────────────────────────────────────────────────

def _route_after_collect(state: Phase1State) -> str:
    if not state.get("scored_filtered_topics"):
        return "finalize_phase1"
    return "run_selector"


# ── Node: run_selector ─────────────────────────────────────────────────────────

async def run_selector(state: Phase1State) -> dict:
    """Call the Selector agent with all viable topics."""
    from src.orchestrators.phase1.registry_client import call_selector

    run_id = state["run_id"]
    topics = state.get("scored_filtered_topics", [])
    aap_token = state.get("aap_token")

    tracer = get_tracer("news-mas-orchestrator-1")
    with tracer.start_as_current_span("phase1.selector") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("topic_count", len(topics))

        agent_token = _exchange_for_agent(aap_token, "selector", ["select.candidates"])
        max_select = int(os.getenv("MAX_TOPICS", "5"))

        try:
            inp = SelectorInput(
                run_id=run_id,
                topics=topics,
                max_select=max_select,
                aap_token=agent_token,
            )
            output = await call_selector(inp)

            span.set_attribute("selected_count", len(output.selected))
            span.set_attribute("rejected_count", len(output.rejected))
            token_est = sum(
                _estimate_tokens(t.topic_text) for t in topics
            ) * 2  # input + output estimate

            if not output.success:
                err = output.error or PipelineError(
                    error_code="SELECTOR_FAILED",
                    severity=ErrorSeverity.FATAL,
                    agent_id="selector",
                    run_id=run_id,
                    message="Selector returned failure",
                )
                logger.error(
                    "selector_failed",
                    extra={"run_id": run_id, "error_code": err.error_code},
                )
                return {
                    "selector_output": output,
                    "run_status": "failed",
                    "errors": [err],
                    "total_tokens_used": token_est,
                }

            logger.info(
                "selector_complete",
                extra={
                    "run_id": run_id,
                    "selected_count": len(output.selected),
                    "selection_confidence": output.selection_confidence,
                },
            )
            return {
                "selector_output": output,
                "total_tokens_used": token_est,
            }

        except Exception as exc:
            err = PipelineError(
                error_code="SELECTOR_EXCEPTION",
                severity=ErrorSeverity.FATAL,
                agent_id="selector",
                run_id=run_id,
                message="Unexpected exception in run_selector node",
                context={"error_type": type(exc).__name__},
            )
            logger.error(
                "run_selector_exception",
                extra={"run_id": run_id, "error_type": type(exc).__name__},
            )
            return {
                "selector_output": None,
                "run_status": "failed",
                "errors": [err],
                "total_tokens_used": 0,
            }


# ── Edge: after selector ───────────────────────────────────────────────────────

def _route_after_selector(state: Phase1State) -> str:
    sel = state.get("selector_output")
    if state.get("run_status") == "failed" or sel is None or not sel.success:
        return "finalize_phase1"
    return "run_phase1_judge"


# ── Node: run_phase1_judge ─────────────────────────────────────────────────────

async def run_phase1_judge(state: Phase1State) -> dict:
    """
    Call the Phase1Judge agent.  On FATAL failure: use selector's choices
    unchanged (DEGRADED mode) — the pipeline must never silently abort at this
    stage since the selector already validated candidates.
    """
    from src.orchestrators.phase1.registry_client import call_phase1_judge

    run_id = state["run_id"]
    selector_out: SelectorOutput = state["selector_output"]
    all_candidates = state.get("scored_filtered_topics", [])
    aap_token = state.get("aap_token")

    tracer = get_tracer("news-mas-orchestrator-1")
    with tracer.start_as_current_span("phase1.judge") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("candidate_count", len(all_candidates))
        span.set_attribute("selected_count", len(selector_out.selected))

        agent_token = _exchange_for_agent(aap_token, "phase1-judge", ["judge.phase1"])

        try:
            inp = Phase1JudgeInput(
                run_id=run_id,
                all_candidates=all_candidates,
                selected=selector_out.selected,
                rejected=selector_out.rejected,
                selected_articles=selector_out.selected_articles,
                since_date=state.get("since_date"),
                aap_token=agent_token,
            )
            output = await call_phase1_judge(inp)

            span.set_attribute("verdict", output.verdict)
            span.set_attribute("adjustment_count", len(output.adjustments))
            token_est = len(all_candidates) * 300  # judge sees all candidates

            if not output.success and output.error:
                sev = output.error.severity
                if sev == ErrorSeverity.FATAL:
                    logger.warning(
                        "judge_unavailable_degraded_fallback",
                        extra={"run_id": run_id, "error_code": output.error.error_code},
                    )
                    # DEGRADED: pipeline continues with selector choices
                    return {
                        "judge_output": output,
                        "total_tokens_used": token_est,
                        "errors": [output.error],
                    }

            logger.info(
                "judge_complete",
                extra={
                    "run_id": run_id,
                    "verdict": output.verdict,
                    "adjustment_count": len(output.adjustments),
                    "overall_confidence": output.overall_confidence,
                },
            )
            return {
                "judge_output": output,
                "total_tokens_used": token_est,
            }

        except Exception as exc:
            err = PipelineError(
                error_code="JUDGE_EXCEPTION",
                severity=ErrorSeverity.DEGRADED,
                agent_id="phase1-judge",
                run_id=run_id,
                message="Unexpected exception in run_phase1_judge node",
                context={"error_type": type(exc).__name__},
            )
            logger.error(
                "run_phase1_judge_exception",
                extra={"run_id": run_id, "error_type": type(exc).__name__},
            )
            # DEGRADED fallback: build a synthetic output from selector choices
            from src.common.schemas import JudgedArticle, JudgedTopic
            fallback_selected = [
                JudgedTopic(topic_id=s.topic_id, rank=s.rank, source="selector")
                for s in sorted(selector_out.selected, key=lambda x: x.rank)
            ]
            fallback_articles = [
                JudgedArticle(article=sa.article, approved=True, reason="selector fallback")
                for sa in selector_out.selected_articles
            ]
            degraded_output = Phase1JudgeOutput(
                run_id=run_id,
                success=True,
                error=err,
                verdict="endorsed",
                final_selected=fallback_selected,
                adjustments=[],
                endorsement_note="Judge exception — selector choices returned unchanged.",
                overall_confidence="LOW",
                judged_articles=fallback_articles,
                warnings=["judge_exception:DEGRADED"],
            )
            return {
                "judge_output": degraded_output,
                "errors": [err],
                "total_tokens_used": 0,
            }


# ── Node: finalize_phase1 ──────────────────────────────────────────────────────

async def finalize_phase1(state: Phase1State) -> dict:
    """
    Build ranked_candidates from judge output, add tombstones, persist final
    run state, emit phase1.complete OTel event.
    """
    run_id = state["run_id"]
    judge_out: Optional[Phase1JudgeOutput] = state.get("judge_output")
    selector_out: Optional[SelectorOutput] = state.get("selector_output")
    all_topics = state.get("topics", [])
    existing_errors = state.get("errors", [])

    tracer = get_tracer("news-mas-orchestrator-1")
    start_time = time.monotonic()

    with tracer.start_as_current_span("phase1.finalize") as span:
        span.set_attribute("run_id", run_id)

        ranked: list[RankedCandidate] = []

        # ── Build ranked candidates from judge (or selector fallback) ────────
        if judge_out and judge_out.final_selected:
            article_map: dict[str, list[JudgedArticle]] = {}
            for ja in judge_out.judged_articles:
                # Group approved articles by approximate origin — we'll match by URL
                pass

            # Build a topic_id → articles mapping from the judge's judged_articles
            # judged_articles is flat; we need to reverse-map to topics
            topic_to_articles: dict[str, list[JudgedArticle]] = {}
            candidate_map = {t.topic_id: t for t in state.get("scored_filtered_topics", [])}
            for jt in judge_out.final_selected:
                cand = candidate_map.get(jt.topic_id)
                if cand is None:
                    continue
                kept_urls = {sa.article.url for sa in cand.kept_results}
                topic_articles = [
                    ja for ja in judge_out.judged_articles
                    if ja.approved and ja.article.url in kept_urls
                ]
                topic_to_articles[jt.topic_id] = topic_articles

            for jt in sorted(judge_out.final_selected, key=lambda x: x.rank):
                # Look up confidence from candidate
                cand = candidate_map.get(jt.topic_id)
                conf = cand.confidence if cand else "LOW"
                ranked.append(RankedCandidate(
                    topic_id=jt.topic_id,
                    rank=jt.rank,
                    source=jt.source,
                    confidence=conf,
                    judged_articles=topic_to_articles.get(jt.topic_id, []),
                ))

        elif selector_out and selector_out.selected:
            # Selector output without judge (e.g., circuit_breaker or budget_exceeded
            # reached after selector ran — unusual but handle gracefully)
            candidate_map = {t.topic_id: t for t in state.get("scored_filtered_topics", [])}
            for st in sorted(selector_out.selected, key=lambda x: x.rank):
                cand = candidate_map.get(st.topic_id)
                conf = cand.confidence if cand else "LOW"
                ranked.append(RankedCandidate(
                    topic_id=st.topic_id,
                    rank=st.rank,
                    source="selector",
                    confidence=conf,
                ))

        # ── Add tombstone entries for failed topics ───────────────────────────
        selected_ids = {c.topic_id for c in ranked}
        tombstone_errors = [e for e in existing_errors if e.topic_id and e.topic_id not in selected_ids]
        tombstoned_ids = {e.topic_id for e in tombstone_errors}

        for err in tombstone_errors:
            if err.topic_id and err.topic_id not in selected_ids:
                ranked.append(RankedCandidate(
                    topic_id=err.topic_id,
                    rank=len(ranked) + 1,
                    is_tombstone=True,
                    tombstone_reason=err.message,
                ))

        # ── Determine final run status ────────────────────────────────────────
        run_status = state.get("run_status", "unknown")
        if run_status not in ("circuit_breaker_active", "budget_exceeded", "no_topics", "failed"):
            if not any(not c.is_tombstone for c in ranked):
                run_status = "no_candidates"
            else:
                run_status = "complete"

        total_tokens = state.get("total_tokens_used", 0)
        candidates_count = sum(1 for c in ranked if not c.is_tombstone)
        tombstone_count = sum(1 for c in ranked if c.is_tombstone)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        span.set_attribute("run_status", run_status)
        span.set_attribute("candidates_count", candidates_count)
        span.set_attribute("tombstone_count", tombstone_count)
        span.set_attribute("total_tokens_used", total_tokens)

        span.add_event(
            "phase1.complete",
            attributes={
                "run_id": run_id,
                "run_status": run_status,
                "candidates_count": candidates_count,
                "tombstone_count": tombstone_count,
                "total_tokens_used": total_tokens,
                "duration_ms": duration_ms,
            },
        )

        # ── Persist final run state ───────────────────────────────────────────
        repo = _try_get_run_repo()
        if repo is not None:
            try:
                repo.save(run_id, {
                    "run_id": run_id,
                    "status": run_status,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "topics": [t.topic_id for t in all_topics],
                    "metrics": {
                        "candidates_count": candidates_count,
                        "tombstone_count": tombstone_count,
                        "total_tokens_used": total_tokens,
                        "duration_ms": duration_ms,
                    },
                })
            except Exception as exc:
                logger.warning(
                    "run_state_final_save_failed",
                    extra={"run_id": run_id, "error_type": type(exc).__name__},
                )

        logger.info(
            "phase1_complete",
            extra={
                "run_id": run_id,
                "run_status": run_status,
                "candidates_count": candidates_count,
                "tombstone_count": tombstone_count,
                "total_tokens_used": total_tokens,
                "duration_ms": duration_ms,
            },
        )

    return {
        "ranked_candidates": ranked,
        "run_status": run_status,
    }


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_phase1_graph():
    """
    Assemble and compile the Phase 1 LangGraph.

    Parallel fan-out structure:
      initialize_run
        → [fan_out_search edge] → search_topic × N (parallel)
        → fan_out_heat_score
        → [_route_heat_score edge] → heat_score_topic × M (parallel)
        → fan_out_filter
        → [_route_filter edge] → filter_topic × K (parallel)
        → collect_phase1_results
        → [_route_after_collect edge] → run_selector OR finalize_phase1
        → [_route_after_selector edge] → run_phase1_judge OR finalize_phase1
        → finalize_phase1
        → END
    """
    g = StateGraph(Phase1State)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("initialize_run", initialize_run)
    g.add_node("search_topic", search_topic)
    g.add_node("fan_out_heat_score", fan_out_heat_score)
    g.add_node("heat_score_topic", heat_score_topic)
    g.add_node("fan_out_filter", fan_out_filter)
    g.add_node("filter_topic", filter_topic)
    g.add_node("collect_phase1_results", collect_phase1_results)
    g.add_node("run_selector", run_selector)
    g.add_node("run_phase1_judge", run_phase1_judge)
    g.add_node("finalize_phase1", finalize_phase1)

    # ── Edges ──────────────────────────────────────────────────────────────────
    g.set_entry_point("initialize_run")

    # Fan-out stage 1: search
    # fan_out_search returns [Send("search_topic", ...)] or the string "finalize_phase1"
    g.add_conditional_edges(
        "initialize_run",
        fan_out_search,
        ["search_topic", "finalize_phase1"],
    )
    # After ALL search_topic complete → fan_out_heat_score (runs once with merged state)
    g.add_edge("search_topic", "fan_out_heat_score")

    # Fan-out stage 2: heat score
    g.add_conditional_edges(
        "fan_out_heat_score",
        _route_heat_score,
        ["heat_score_topic", "collect_phase1_results", "finalize_phase1"],
    )
    # After ALL heat_score_topic complete → fan_out_filter (once)
    g.add_edge("heat_score_topic", "fan_out_filter")

    # Fan-out stage 3: filter
    g.add_conditional_edges(
        "fan_out_filter",
        _route_filter,
        ["filter_topic", "collect_phase1_results", "finalize_phase1"],
    )
    # After ALL filter_topic complete → collect_phase1_results (once)
    g.add_edge("filter_topic", "collect_phase1_results")

    # Sequential stage: selector
    g.add_conditional_edges(
        "collect_phase1_results",
        _route_after_collect,
        ["run_selector", "finalize_phase1"],
    )

    # Sequential stage: judge
    g.add_conditional_edges(
        "run_selector",
        _route_after_selector,
        ["run_phase1_judge", "finalize_phase1"],
    )

    g.add_edge("run_phase1_judge", "finalize_phase1")
    g.add_edge("finalize_phase1", END)

    return g.compile()


# ── Module-level compiled graph ────────────────────────────────────────────────

phase1_graph = build_phase1_graph()
