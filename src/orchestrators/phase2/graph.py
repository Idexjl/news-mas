"""
Phase 2 LangGraph orchestrator.

Pipeline: Summarizer ⇄ Reviewer (retry loop, max N) → RelevanceGate → Digest

Parallel fan-out
----------------
Each candidate is processed independently via LangGraph Send() — one
process_candidate node per candidate, all running concurrently. The retry
loop runs locally within each process_candidate invocation; there is no
cross-candidate shared state during processing.

Retry loop (per candidate)
--------------------------
  summarize → review →
    if pass: gate
    if fail + retries remaining: retry_summarize (with feedback) → review
    if fail + budget exhausted: gate (with budget_exhausted=True)

  Summarizer FATAL error → tombstone immediately (no gate call)
  Reviewer unavailable → proceed unreviewed (DEGRADED)
  Gate error → default pass (DEGRADED)

Token budget
------------
Phase 1 token usage is carried forward in phase1_tokens_used. If Phase 1
already consumed the full budget, all Phase 2 candidates are tombstoned at
initialize_phase2 before any processing starts. Each process_candidate also
estimates its own token cost and tombstones if that estimate plus phase1
usage already exceeds MAX_TOKENS_PER_RUN.

[DPOP-TODO] All agent calls in process_candidate should use DPoP-bound
Entra ID tokens once the auth layer is active. See phase1/graph.py
[DPOP-TODO] block for the full template.
"""
from __future__ import annotations

import operator
import os
import time
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.auth.token_service import exchange_token, mint_token
from src.common.error_codes import BUDGET_EXHAUSTED
from src.common.observability import get_logger, get_tracer, setup_telemetry
from src.common.pipeline_errors import ErrorSeverity, PipelineError
from src.common.schemas import (
    DigestEntry,
    DigestOutput,
    DigestTombstone,
    RankedCandidate,
    RelevanceGateInput,
    RelevanceGateOutput,
    ReviewerInput,
    ReviewerOutput,
    SearchResult,
    SummarizerInput,
    SummarizerOutput,
)

load_dotenv()
setup_telemetry("news-mas-orchestrator-2")
logger = get_logger(__name__)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
_MAX_TOKENS_DEFAULT = 2_000_000
_CONFIDENCE_ORDER: dict[str, int] = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ── State reducers ─────────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


# ── Phase2State ────────────────────────────────────────────────────────────────

class Phase2State(TypedDict):
    run_id: str
    user_id: str
    ranked_candidates: list[RankedCandidate]
    # Annotated fields: parallel process_candidate nodes write keyed by topic_id
    summaries: Annotated[dict[str, SummarizerOutput], _merge_dicts]
    reviewer_verdicts: Annotated[dict[str, ReviewerOutput], _merge_dicts]
    gate_decisions: Annotated[dict[str, RelevanceGateOutput], _merge_dicts]
    retry_counts: Annotated[dict[str, int], _merge_dicts]
    budget_exhausted: Annotated[dict[str, bool], _merge_dicts]
    errors: Annotated[list[PipelineError], operator.add]
    total_tokens_used: Annotated[int, operator.add]
    # Fields set by single nodes
    digest_entries: list[DigestEntry]
    tombstones: list[DigestTombstone]
    run_status: str
    aap_token: Optional[str]
    phase1_tokens_used: int  # Phase 1 token budget already consumed


# ── Helpers ────────────────────────────────────────────────────────────────────

def _max_tokens() -> int:
    return int(os.getenv("MAX_TOKENS_PER_RUN", str(_MAX_TOKENS_DEFAULT)))


def _secret() -> str:
    return os.getenv("MAS_SECRET_KEY", "")


def _mint_orchestrator_token(run_id: str, user_id: str) -> Optional[str]:
    secret = _secret()
    if not secret:
        return None
    try:
        return mint_token(
            subject=f"orchestrator:{user_id}",
            agent_name="phase2-orchestrator",
            model_provider="none",
            model_id="none",
            capabilities=["orchestrate.phase2"],
            task_id=run_id,
            task_description="Phase 2 summarization and quality gate pipeline",
            data_sensitivity="standard",
            oversight_required=False,
            oversight_level="none",
            context={"run_id": run_id, "user_id": user_id},
            secret_key=secret,
        )
    except Exception as exc:
        logger.warning("aap_token_mint_failed", extra={"error_type": type(exc).__name__})
        return None


_PHASE2_MODEL_PROVIDERS: dict[str, tuple[str, str]] = {
    "summarizer": ("anthropic", "claude-sonnet-4-6"),
    "reviewer": ("anthropic", "claude-sonnet-4-6"),
    "relevance-gate": ("anthropic", "claude-haiku-4-5-20251001"),
}


def _exchange_for_agent(
    token: Optional[str], agent_name: str, capabilities: list[str]
) -> Optional[str]:
    if not token:
        return None
    secret = _secret()
    if not secret:
        return None
    provider, model_id = _PHASE2_MODEL_PROVIDERS.get(agent_name, ("none", "none"))
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


def _try_get_repos():
    """Return (RunRepository, DigestRepository) or (None, None) when storage unavailable."""
    try:
        from src.common.storage import DigestRepository, FernetStorage, RunRepository
        storage = FernetStorage()
        return RunRepository(storage), DigestRepository(storage)
    except Exception:
        return None, None


def _gate_tombstone(
    run_id: str,
    candidate: RankedCandidate,
    reason_code: str,
    message: str,
) -> RelevanceGateOutput:
    """Build a RelevanceGateOutput that carries a tombstone for this candidate."""
    tombstone = DigestTombstone(
        topic_id=candidate.topic_id,
        topic_display_name=candidate.topic_text or candidate.topic_id,
        tombstone_message=message,
        heat_score=candidate.heat_score,
        confidence=candidate.confidence,
        reason_code=reason_code,  # type: ignore[arg-type]
    )
    return RelevanceGateOutput(
        run_id=run_id,
        success=True,
        decision="tombstone",
        gate_path="fast_tombstone",
        tombstone=tombstone,
    )


def _estimate_candidate_tokens(candidate: RankedCandidate) -> int:
    """Rough per-candidate token budget estimate across summarizer + reviewer + gate."""
    source_chars = sum(len(ja.article.content or "") for ja in candidate.judged_articles)
    return (source_chars // 4) * 3 + 1500  # 3× for LLM I/O, 1500 base overhead


# ── Node: initialize_phase2 ────────────────────────────────────────────────────

async def initialize_phase2(state: Phase2State) -> dict:
    run_id = state["run_id"]
    tracer = get_tracer("news-mas-orchestrator-2")

    with tracer.start_as_current_span("phase2.initialize") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("candidate_count", len(state["ranked_candidates"]))

        aap_token = _mint_orchestrator_token(run_id, state["user_id"])

        phase1_tokens = state.get("phase1_tokens_used", 0)
        budget = _max_tokens()

        if phase1_tokens >= budget:
            logger.warning(
                "phase2_budget_already_exceeded",
                extra={"run_id": run_id, "phase1_tokens": phase1_tokens, "budget": budget},
            )
            span.set_attribute("budget_exceeded_at_start", True)
            span.add_event(
                "phase2.budget_exceeded",
                attributes={"run_id": run_id, "phase1_tokens": phase1_tokens, "budget": budget},
            )
            return {"run_status": "budget_exceeded", "aap_token": aap_token}

        logger.info(
            "phase2_initialized",
            extra={
                "run_id": run_id,
                "candidate_count": len(state["ranked_candidates"]),
                "aap_token_minted": aap_token is not None,
                "phase1_tokens_used": phase1_tokens,
            },
        )
        return {"run_status": "running", "aap_token": aap_token}


# ── Fan-out routing ────────────────────────────────────────────────────────────

def fan_out_summarize(state: Phase2State):
    """
    Conditional edge from initialize_phase2.
    Returns [Send("process_candidate", ...)] per candidate, or "build_digest" on early exit.
    """
    if state["run_status"] == "budget_exceeded":
        return "build_digest"

    candidates = state.get("ranked_candidates", [])
    if not candidates:
        return "build_digest"

    phase1_tokens = state.get("phase1_tokens_used", 0)

    return [
        Send("process_candidate", {
            "candidate": c,
            "run_id": state["run_id"],
            "aap_token": state["aap_token"],
            "phase1_tokens_used": phase1_tokens,
        })
        for c in candidates
    ]


# ── Node: process_candidate (parallel, one per candidate) ─────────────────────

async def process_candidate(state: dict) -> dict:
    """
    Process one candidate through the summarize → review → gate pipeline.
    Handles the retry loop internally. Runs concurrently with other candidates.

    Returns dict updates that are merged into Phase2State via Annotated reducers.
    """
    from src.agents.relevance_gate.agent import run_agent as run_gate_agent
    from src.agents.reviewer.agent import run_agent as run_reviewer_agent
    from src.agents.summarizer.agent import run_agent as run_summarizer_agent

    candidate: RankedCandidate = state["candidate"]
    run_id: str = state["run_id"]
    aap_token: Optional[str] = state.get("aap_token")
    phase1_tokens: int = state.get("phase1_tokens_used", 0)
    topic_id = candidate.topic_id

    tracer = get_tracer("news-mas-orchestrator-2")

    with tracer.start_as_current_span(f"phase2.candidate.{topic_id}") as cand_span:
        cand_span.set_attribute("topic_id", topic_id)
        cand_span.set_attribute("run_id", run_id)

        # ── Phase 1 tombstone passthrough ──────────────────────────────────────
        if candidate.is_tombstone:
            gate_out = _gate_tombstone(
                run_id, candidate, "gate_decision",
                candidate.tombstone_reason or "Topic excluded in Phase 1.",
            )
            cand_span.set_attribute("final_verdict", "tombstone")
            cand_span.set_attribute("gate_decision", "tombstone")
            cand_span.set_attribute("retry_count", 0)
            return {
                "gate_decisions": {topic_id: gate_out},
                "retry_counts": {topic_id: 0},
                "budget_exhausted": {topic_id: False},
                "errors": [],
                "total_tokens_used": 0,
            }

        # ── Per-candidate token budget pre-check ───────────────────────────────
        estimated_tokens = _estimate_candidate_tokens(candidate)
        if phase1_tokens + estimated_tokens > _max_tokens():
            logger.warning(
                "phase2_candidate_budget_tombstone",
                extra={"run_id": run_id, "topic_id": topic_id,
                       "estimated_tokens": estimated_tokens},
            )
            gate_out = _gate_tombstone(
                run_id, candidate, "budget_exhausted",
                "Coverage for this topic was not completed within the refresh window.",
            )
            cand_span.set_attribute("final_verdict", "tombstone")
            cand_span.set_attribute("gate_decision", "tombstone")
            cand_span.set_attribute("budget_exhausted", True)
            cand_span.set_attribute("retry_count", 0)
            return {
                "gate_decisions": {topic_id: gate_out},
                "retry_counts": {topic_id: 0},
                "budget_exhausted": {topic_id: True},
                "errors": [PipelineError(
                    error_code=BUDGET_EXHAUSTED,
                    severity=ErrorSeverity.SKIP,
                    agent_id="phase2-orchestrator",
                    run_id=run_id,
                    topic_id=topic_id,
                    message="Token budget exceeded; candidate tombstoned without processing",
                    context={"estimated_tokens": estimated_tokens,
                             "phase1_tokens": phase1_tokens},
                )],
                "total_tokens_used": 0,
            }

        # ── Build sources from judged articles ─────────────────────────────────
        approved = [ja for ja in candidate.judged_articles if ja.approved]
        if not approved:
            gate_out = _gate_tombstone(
                run_id, candidate, "gate_decision",
                "No approved articles were available for this topic.",
            )
            cand_span.set_attribute("final_verdict", "tombstone")
            cand_span.set_attribute("gate_decision", "tombstone")
            cand_span.set_attribute("retry_count", 0)
            return {
                "gate_decisions": {topic_id: gate_out},
                "retry_counts": {topic_id: 0},
                "budget_exhausted": {topic_id: False},
                "errors": [],
                "total_tokens_used": 0,
            }

        primary_article = approved[0].article
        sources = [
            SearchResult(
                url=ja.article.url,
                title=ja.article.title,
                content=ja.article.content,
                source=ja.article.source,
                published_at=ja.article.published_at,
            )
            for ja in approved
        ]

        # ── Retry loop ─────────────────────────────────────────────────────────
        retry_count = 0
        reviewer_feedback: Optional[str] = None
        summary_out: Optional[SummarizerOutput] = None
        review_out: Optional[ReviewerOutput] = None
        budget_exhausted_flag = False
        total_tokens = 0
        errors: list[PipelineError] = []

        with tracer.start_as_current_span(f"phase2.retry_loop.{topic_id}") as loop_span:
            loop_span.set_attribute("max_retries", MAX_RETRIES)

            while retry_count <= MAX_RETRIES:

                # ── Summarize ──────────────────────────────────────────────────
                summarizer_token = _exchange_for_agent(
                    aap_token, "summarizer", ["summarize.content"]
                )
                sum_inp = SummarizerInput(
                    run_id=run_id,
                    article=primary_article,
                    sources=sources,
                    topic_text=candidate.topic_text,
                    reviewer_feedback=reviewer_feedback,
                    retry_count=retry_count,
                    aap_token=summarizer_token,
                )

                try:
                    summary_out = await run_summarizer_agent(sum_inp)
                    total_tokens += max(500, estimated_tokens // 3)
                except Exception as exc:
                    fatal_err = PipelineError(
                        error_code="SUMMARIZER_EXCEPTION",
                        severity=ErrorSeverity.FATAL,
                        agent_id="summarizer",
                        run_id=run_id,
                        topic_id=topic_id,
                        message="Summarizer raised unexpected exception",
                        context={"error_type": type(exc).__name__,
                                 "retry_count": retry_count},
                    )
                    errors.append(fatal_err)
                    logger.error(
                        "process_candidate_summarizer_exception",
                        extra={"run_id": run_id, "topic_id": topic_id,
                               "error_type": type(exc).__name__},
                    )
                    gate_out = _gate_tombstone(
                        run_id, candidate, "gate_decision",
                        "Coverage of this topic could not be completed.",
                    )
                    loop_span.set_attribute("actual_retries", retry_count)
                    loop_span.set_attribute("budget_exhausted", False)
                    cand_span.set_attribute("final_verdict", "tombstone")
                    cand_span.set_attribute("gate_decision", "tombstone")
                    cand_span.set_attribute("retry_count", retry_count)
                    return {
                        "gate_decisions": {topic_id: gate_out},
                        "retry_counts": {topic_id: retry_count},
                        "budget_exhausted": {topic_id: False},
                        "errors": errors,
                        "total_tokens_used": total_tokens,
                    }

                if not summary_out.success:
                    err = summary_out.error
                    if err and err.severity == ErrorSeverity.FATAL:
                        errors.append(err)
                        gate_out = _gate_tombstone(
                            run_id, candidate, "gate_decision",
                            "Coverage of this topic could not be completed.",
                        )
                        loop_span.set_attribute("actual_retries", retry_count)
                        loop_span.set_attribute("budget_exhausted", False)
                        cand_span.set_attribute("final_verdict", "tombstone")
                        cand_span.set_attribute("gate_decision", "tombstone")
                        cand_span.set_attribute("retry_count", retry_count)
                        return {
                            "gate_decisions": {topic_id: gate_out},
                            "retry_counts": {topic_id: retry_count},
                            "budget_exhausted": {topic_id: False},
                            "errors": errors,
                            "total_tokens_used": total_tokens,
                        }
                    # RETRYABLE — try again
                    if err:
                        errors.append(err)
                    retry_count += 1
                    continue

                # ── Review ─────────────────────────────────────────────────────
                reviewer_token = _exchange_for_agent(
                    aap_token, "reviewer", ["review.content"]
                )
                rev_inp = ReviewerInput(
                    run_id=run_id,
                    article=primary_article,
                    summary=summary_out.summary,
                    key_points=summary_out.key_points,
                    citations=summary_out.citations,
                    sources=sources,
                    retry_count=retry_count,
                    aap_token=reviewer_token,
                )

                try:
                    review_out = await run_reviewer_agent(rev_inp)
                    total_tokens += max(300, estimated_tokens // 4)
                except Exception as exc:
                    logger.warning(
                        "phase2_reviewer_exception_degraded",
                        extra={"run_id": run_id, "topic_id": topic_id,
                               "error_type": type(exc).__name__},
                    )
                    review_out = ReviewerOutput(
                        run_id=run_id,
                        success=True,
                        approved=True,
                        verdict="pass",
                        confidence="LOW",
                        endorsement_note="reviewer_exception — proceeding unreviewed",
                        warnings=["reviewer_exception"],
                    )
                    break

                if not review_out.success:
                    # Reviewer unavailable (DEGRADED) — proceed unreviewed
                    break

                if review_out.verdict == "pass":
                    break  # proceed to gate

                # Failed review — consume a retry slot
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    budget_exhausted_flag = True
                    break

                reviewer_feedback = review_out.feedback

            # ── End of retry loop ──────────────────────────────────────────────
            loop_span.set_attribute("actual_retries", retry_count)
            loop_span.set_attribute("budget_exhausted", budget_exhausted_flag)

        # ── Relevance gate ─────────────────────────────────────────────────────
        reviewer_verdict: Literal["pass", "fail"] = "pass"
        if review_out is not None:
            reviewer_verdict = review_out.verdict  # type: ignore[assignment]

        gate_token = _exchange_for_agent(aap_token, "relevance-gate", ["gate.relevance"])
        gate_inp = RelevanceGateInput(
            run_id=run_id,
            topic_id=topic_id,
            topic_text=candidate.topic_text,
            heat_score=candidate.heat_score,
            overall_confidence=candidate.confidence,
            reviewer_verdict=reviewer_verdict,
            retry_count=retry_count,
            budget_exhausted=budget_exhausted_flag,
            tombstone_reason=(
                "Budget exhausted after maximum retries" if budget_exhausted_flag else None
            ),
            reviewer_last_feedback=reviewer_feedback,
            aap_token=gate_token,
        )

        try:
            gate_out = await run_gate_agent(gate_inp)
            total_tokens += 200
        except Exception as exc:
            logger.warning(
                "phase2_gate_exception_defaulting_pass",
                extra={"run_id": run_id, "topic_id": topic_id,
                       "error_type": type(exc).__name__},
            )
            gate_out = RelevanceGateOutput(
                run_id=run_id,
                success=True,
                decision="pass",
                gate_path="fast_pass",
            )
            errors.append(PipelineError(
                error_code="GATE_EXCEPTION",
                severity=ErrorSeverity.DEGRADED,
                agent_id="relevance-gate",
                run_id=run_id,
                topic_id=topic_id,
                message="Gate raised unexpected exception; defaulting to pass",
                context={"error_type": type(exc).__name__},
            ))

        cand_span.set_attribute("final_verdict",
                                review_out.verdict if review_out else "pass")
        cand_span.set_attribute("gate_decision", gate_out.decision)
        cand_span.set_attribute("retry_count", retry_count)

        result: dict = {
            "gate_decisions": {topic_id: gate_out},
            "retry_counts": {topic_id: retry_count},
            "budget_exhausted": {topic_id: budget_exhausted_flag},
            "errors": errors,
            "total_tokens_used": total_tokens,
        }

        if summary_out is not None and summary_out.success:
            result["summaries"] = {topic_id: summary_out}

        if review_out is not None:
            result["reviewer_verdicts"] = {topic_id: review_out}

        return result


# ── Node: collect_phase2_results ───────────────────────────────────────────────

async def collect_phase2_results(state: Phase2State) -> dict:
    """
    Waypoint after all parallel process_candidate nodes complete.
    Checks total token budget and logs metrics. State already merged by reducers.
    """
    run_id = state["run_id"]
    total_tokens = state.get("total_tokens_used", 0)
    phase1_tokens = state.get("phase1_tokens_used", 0)
    combined = phase1_tokens + total_tokens
    budget = _max_tokens()

    if combined > budget:
        tracer = get_tracer("news-mas-orchestrator-2")
        with tracer.start_as_current_span("phase2.budget_exceeded_event") as span:
            span.add_event(
                "phase2.budget_exceeded",
                attributes={"run_id": run_id, "total_tokens_used": combined,
                            "budget": budget},
            )
        logger.warning(
            "phase2_total_budget_exceeded",
            extra={"run_id": run_id, "total_tokens_used": combined, "budget": budget},
        )

    logger.info(
        "phase2_results_collected",
        extra={
            "run_id": run_id,
            "summaries_count": len(state.get("summaries", {})),
            "gate_decisions_count": len(state.get("gate_decisions", {})),
            "total_tokens_used": total_tokens,
        },
    )
    return {}


# ── Node: build_digest ─────────────────────────────────────────────────────────

async def build_digest(state: Phase2State) -> dict:
    """
    Assemble digest entries and tombstones from gate decisions and summaries.

    Sorting: Phase 1 rank (primary) → heat score descending (secondary)
             → confidence descending (tertiary).
    Tombstones appended after all passing entries.
    """
    run_id = state["run_id"]
    ranked_candidates = state.get("ranked_candidates", [])
    summaries = state.get("summaries", {})
    gate_decisions = state.get("gate_decisions", {})

    tracer = get_tracer("news-mas-orchestrator-2")

    with tracer.start_as_current_span("phase2.digest_assembly") as span:
        span.set_attribute("run_id", run_id)

        entries_with_rank: list[tuple[int, float, int, DigestEntry]] = []
        tombstones: list[DigestTombstone] = []

        for candidate in sorted(ranked_candidates, key=lambda c: c.rank):
            topic_id = candidate.topic_id
            gate_out: Optional[RelevanceGateOutput] = gate_decisions.get(topic_id)

            if gate_out is None:
                tombstones.append(DigestTombstone(
                    topic_id=topic_id,
                    topic_display_name=candidate.topic_text or topic_id,
                    tombstone_message="Processing was not completed for this topic.",
                    heat_score=candidate.heat_score,
                    confidence=candidate.confidence,
                    reason_code="gate_decision",
                ))
                continue

            if gate_out.decision == "tombstone":
                if gate_out.tombstone:
                    tombstones.append(gate_out.tombstone)
                else:
                    tombstones.append(DigestTombstone(
                        topic_id=topic_id,
                        topic_display_name=candidate.topic_text or topic_id,
                        tombstone_message="This topic was not included this week.",
                        heat_score=candidate.heat_score,
                        confidence=candidate.confidence,
                        reason_code="gate_decision",
                    ))
                continue

            # Gate passed — build a digest entry from the summary
            summary_out: Optional[SummarizerOutput] = summaries.get(topic_id)
            if summary_out is None or not summary_out.success:
                tombstones.append(DigestTombstone(
                    topic_id=topic_id,
                    topic_display_name=candidate.topic_text or topic_id,
                    tombstone_message="Summary was not available for this topic.",
                    heat_score=candidate.heat_score,
                    confidence=candidate.confidence,
                    reason_code="gate_decision",
                ))
                continue

            primary_article = next(
                (ja.article for ja in candidate.judged_articles if ja.approved), None
            )
            if primary_article is None:
                tombstones.append(DigestTombstone(
                    topic_id=topic_id,
                    topic_display_name=candidate.topic_text or topic_id,
                    tombstone_message="No articles available for this topic.",
                    heat_score=candidate.heat_score,
                    confidence=candidate.confidence,
                    reason_code="gate_decision",
                ))
                continue

            conf = summary_out.confidence or candidate.confidence
            entry = DigestEntry(
                article=primary_article,
                summary=summary_out.summary,
                key_points=summary_out.key_points,
                citations=summary_out.citations,
                heat_score=candidate.heat_score,
                relevance_confidence=candidate.heat_score,
                confidence=conf,
            )
            entries_with_rank.append((
                candidate.rank,
                -candidate.heat_score,                     # negate for ascending sort
                -_CONFIDENCE_ORDER.get(conf or "LOW", 1),  # negate for ascending sort
                entry,
            ))

        # Sort: rank asc, heat desc, confidence desc
        entries_with_rank.sort(key=lambda x: (x[0], x[1], x[2]))
        digest_entries = [e for _, _, _, e in entries_with_rank]

        span.set_attribute("entry_count", len(digest_entries))
        span.set_attribute("tombstone_count", len(tombstones))

    return {"digest_entries": digest_entries, "tombstones": tombstones}


# ── Node: finalize_phase2 ──────────────────────────────────────────────────────

async def finalize_phase2(state: Phase2State) -> dict:
    """
    Compute overall_confidence, determine run status, persist to storage,
    and emit the phase2.complete OTel event.
    """
    run_id = state["run_id"]
    start = time.monotonic()
    digest_entries = state.get("digest_entries", [])
    tombstones = state.get("tombstones", [])
    errors = state.get("errors", [])
    total_tokens = state.get("total_tokens_used", 0)
    ranked_candidates = state.get("ranked_candidates", [])

    passed_count = len(digest_entries)
    tombstoned_count = len(tombstones)
    total_topics = len(ranked_candidates)
    duration_ms = int((time.monotonic() - start) * 1000)

    # Overall confidence: average of passing entry confidence levels
    if digest_entries:
        scores = [_CONFIDENCE_ORDER.get(e.confidence or "LOW", 1) for e in digest_entries]
        avg = sum(scores) / len(scores)
        if avg >= 2.5:
            overall_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
        elif avg >= 1.5:
            overall_confidence = "MEDIUM"
        else:
            overall_confidence = "LOW"
    else:
        overall_confidence = "LOW"

    if passed_count == 0:
        run_status = "all_tombstoned"
    elif errors:
        run_status = "complete_degraded"
    else:
        run_status = "complete"

    tracer = get_tracer("news-mas-orchestrator-2")
    with tracer.start_as_current_span("phase2.run") as run_span:
        run_span.set_attribute("run_id", run_id)
        run_span.set_attribute("candidate_count", total_topics)
        run_span.set_attribute("passed_count", passed_count)
        run_span.set_attribute("tombstoned_count", tombstoned_count)
        run_span.set_attribute("total_tokens", total_tokens)
        run_span.set_attribute("duration_ms", duration_ms)

        run_span.add_event(
            "phase2.complete",
            attributes={
                "run_id": run_id,
                "run_status": run_status,
                "passed_count": passed_count,
                "tombstoned_count": tombstoned_count,
                "total_tokens_used": total_tokens,
                "duration_ms": duration_ms,
                "overall_confidence": overall_confidence,
            },
        )

    run_repo, digest_repo = _try_get_repos()
    if run_repo is not None:
        try:
            run_repo.save(run_id, {
                "run_id": run_id,
                "status": run_status,
                "phase": 2,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "passed_count": passed_count,
                    "tombstoned_count": tombstoned_count,
                    "total_tokens_used": total_tokens,
                    "duration_ms": duration_ms,
                },
            })
        except Exception as exc:
            logger.warning(
                "phase2_run_save_failed",
                extra={"run_id": run_id, "error_type": type(exc).__name__},
            )

    if digest_repo is not None:
        try:
            digest_repo.save(run_id, {
                "run_id": run_id,
                "user_id": state.get("user_id", ""),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "digest_entries": [e.model_dump(mode="json") for e in digest_entries],
                "tombstones": [t.model_dump(mode="json") for t in tombstones],
                "passed_count": passed_count,
                "tombstoned_count": tombstoned_count,
                "total_tokens_used": total_tokens,
                "run_status": run_status,
                "overall_confidence": overall_confidence,
            })
        except Exception as exc:
            logger.warning(
                "phase2_digest_save_failed",
                extra={"run_id": run_id, "error_type": type(exc).__name__},
            )

    logger.info(
        "phase2_complete",
        extra={
            "run_id": run_id,
            "run_status": run_status,
            "passed_count": passed_count,
            "tombstoned_count": tombstoned_count,
            "total_tokens_used": total_tokens,
            "duration_ms": duration_ms,
            "overall_confidence": overall_confidence,
        },
    )

    return {"run_status": run_status}


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_phase2_graph():
    """
    Assemble and compile the Phase 2 LangGraph.

    Structure:
      initialize_phase2
        → [fan_out_summarize edge] → process_candidate × N (parallel)
        → collect_phase2_results (waypoint — after all parallel complete)
        → build_digest
        → finalize_phase2
        → END

    Early exit (budget_exceeded / no candidates) skips directly to build_digest.
    """
    g = StateGraph(Phase2State)

    g.add_node("initialize_phase2", initialize_phase2)
    g.add_node("process_candidate", process_candidate)
    g.add_node("collect_phase2_results", collect_phase2_results)
    g.add_node("build_digest", build_digest)
    g.add_node("finalize_phase2", finalize_phase2)

    g.set_entry_point("initialize_phase2")

    g.add_conditional_edges(
        "initialize_phase2",
        fan_out_summarize,
        ["process_candidate", "build_digest"],
    )

    # After ALL process_candidate instances complete → collect_phase2_results (once)
    g.add_edge("process_candidate", "collect_phase2_results")
    g.add_edge("collect_phase2_results", "build_digest")
    g.add_edge("build_digest", "finalize_phase2")
    g.add_edge("finalize_phase2", END)

    return g.compile()


# Module-level compiled graph
phase2_graph = build_phase2_graph()
