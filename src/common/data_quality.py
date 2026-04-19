"""
Data quality validators for each pipeline stage.

All methods return PipelineError | None — None means the check passed.
Callers decide whether to surface the error, degrade, or abort based on
error.severity.

PHI safety:
  context dicts are metadata-only. Never pass content values as context.
  The validator enforces this by only recording counts, lengths, and IDs.
"""
from __future__ import annotations

from typing import Any

from src.common.error_codes import (
    ALL_FILTERED,
    CITATION_INVALID,
    CONSTRAINT_VIOLATED,
    HEAT_SCORE_TOO_LOW,
    INSUFFICIENT_RESULTS,
    NO_RESULTS,
    NO_VIABLE_CANDIDATES,
    SUMMARY_TOO_LONG,
    SUMMARY_TOO_SHORT,
)
from src.common.pipeline_errors import ErrorSeverity, PipelineError


class DataQualityValidator:
    """
    Validates data at each pipeline stage boundary.

    agent_id identifies the validator in error records; set it to the
    agent that owns the validation (e.g. "search-worker", "heat-scorer").
    """

    def __init__(self, agent_id: str = "data-quality") -> None:
        self.agent_id = agent_id

    # ── Search / retrieval ────────────────────────────────────────────────────

    def validate_search_results(
        self,
        results: list[Any],
        *,
        min_results: int,
        topic_id: str,
        run_id: str,
    ) -> PipelineError | None:
        """
        Check that Tavily returned usable results.

        Empty → SKIP (no point retrying immediately; may be a dead topic).
        Below threshold → DEGRADED (continue with fewer results).
        """
        if not results:
            return PipelineError(
                error_code=NO_RESULTS,
                severity=ErrorSeverity.SKIP,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message="Search returned no results for topic",
                retry_hint="Try broadening the query or increasing since_days",
                context={
                    "topic_id": topic_id,
                    "min_required": min_results,
                    "result_count": 0,
                },
            )
        if len(results) < min_results:
            return PipelineError(
                error_code=INSUFFICIENT_RESULTS,
                severity=ErrorSeverity.DEGRADED,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=(
                    f"Fewer results than minimum threshold: "
                    f"got {len(results)}, need {min_results}"
                ),
                retry_hint=(
                    "Pipeline continues in degraded mode with available results"
                ),
                context={
                    "topic_id": topic_id,
                    "result_count": len(results),
                    "min_required": min_results,
                },
            )
        return None

    def validate_filtered_results(
        self,
        *,
        filtered_count: int,
        original_count: int,
        run_id: str,
        topic_id: str = "",
    ) -> PipelineError | None:
        """
        Check that the filter agent kept at least one result.

        ALL_FILTERED → SKIP: the topic yielded no articles above the heat
        threshold. Skip this topic; the orchestrator continues with others.
        """
        if original_count > 0 and filtered_count == 0:
            return PipelineError(
                error_code=ALL_FILTERED,
                severity=ErrorSeverity.SKIP,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message="All search results were dropped by the filter agent",
                retry_hint=(
                    "Lower min_heat_score threshold or widen the search query"
                ),
                context={
                    "topic_id": topic_id,
                    "original_count": original_count,
                    "filtered_count": filtered_count,
                },
            )
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def validate_heat_score(
        self,
        score: float,
        *,
        topic_id: str,
        run_id: str,
    ) -> PipelineError | None:
        """
        Reject articles with a zero or negative heat score.

        These articles did not meet any quality signal and should not
        consume summarisation budget.
        """
        if score <= 0.0:
            return PipelineError(
                error_code=HEAT_SCORE_TOO_LOW,
                severity=ErrorSeverity.SKIP,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message="Article heat score is zero — does not meet minimum quality bar",
                retry_hint="Review scoring thresholds or source quality",
                context={
                    "topic_id": topic_id,
                    "heat_score": score,
                },
            )
        return None

    # ── Candidate pool ────────────────────────────────────────────────────────

    def validate_candidates(
        self,
        candidates: list[Any],
        *,
        min_candidates: int,
        run_id: str,
        topic_id: str = "",
    ) -> PipelineError | None:
        """
        Verify the candidate pool is large enough to produce a digest.

        FATAL: no candidates means the pipeline cannot produce any output.
        """
        count = len(candidates) if candidates else 0
        if count < min_candidates:
            return PipelineError(
                error_code=NO_VIABLE_CANDIDATES,
                severity=ErrorSeverity.FATAL,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=(
                    f"Insufficient viable candidates after scoring and filtering: "
                    f"got {count}, need {min_candidates}"
                ),
                retry_hint=(
                    "No articles survived the pipeline — run cannot produce a digest. "
                    "Widen topics, lower heat threshold, or extend since_days."
                ),
                context={
                    "candidate_count": count,
                    "min_required": min_candidates,
                },
            )
        return None

    # ── Summarisation quality ─────────────────────────────────────────────────

    def validate_summary(
        self,
        summary: str,
        *,
        min_length: int,
        max_length: int,
        topic_id: str,
        run_id: str,
    ) -> PipelineError | None:
        """
        Check that a summary meets length constraints.

        Both violations are RETRYABLE: the summariser can be re-prompted
        with explicit length guidance.
        """
        length = len(summary)
        if length < min_length:
            return PipelineError(
                error_code=SUMMARY_TOO_SHORT,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=f"Summary too short: {length} chars (minimum {min_length})",
                retry_hint=(
                    f"Re-prompt summariser: require at least {min_length} characters"
                ),
                context={
                    "topic_id": topic_id,
                    "summary_length": length,
                    "min_length": min_length,
                },
            )
        if length > max_length:
            return PipelineError(
                error_code=SUMMARY_TOO_LONG,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=f"Summary too long: {length} chars (maximum {max_length})",
                retry_hint=(
                    f"Re-prompt summariser: limit to {max_length} characters"
                ),
                context={
                    "topic_id": topic_id,
                    "summary_length": length,
                    "max_length": max_length,
                },
            )
        return None

    # ── Citation integrity ────────────────────────────────────────────────────

    def validate_citations(
        self,
        citations: list[str],
        *,
        sources: list[str],
        topic_id: str,
        run_id: str,
    ) -> PipelineError | None:
        """
        Verify every citation URL is present in the list of actual source URLs.

        Invalid citations indicate hallucination. RETRYABLE: the summariser
        can be re-prompted with the explicit source list.

        Only counts and URL presence are recorded — never URL content or path.
        """
        source_set = set(sources)
        invalid_count = sum(1 for c in citations if c not in source_set)
        if invalid_count:
            return PipelineError(
                error_code=CITATION_INVALID,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=(
                    f"{invalid_count} of {len(citations)} citation(s) "
                    "do not match any provided source URL"
                ),
                retry_hint=(
                    "Re-run summariser with explicit instruction to cite only "
                    "the provided source URLs"
                ),
                context={
                    "topic_id": topic_id,
                    "total_citations": len(citations),
                    "invalid_count": invalid_count,
                    "source_count": len(sources),
                },
            )
        return None

    # ── Constraint adherence ──────────────────────────────────────────────────

    def validate_constraints_respected(
        self,
        content: str,
        *,
        constraints: list[str],
        topic_id: str,
        run_id: str,
    ) -> PipelineError | None:
        """
        Check that required constraint terms appear in the output content.

        This is a structural keyword-presence check — a necessary but not
        sufficient condition for constraint adherence. A full semantic check
        (embedding similarity against the constraint intent) is deferred
        until the offline eval framework is in place (see ARCHITECTURE.md §12).

        RETRYABLE: re-prompting with explicit constraint reinforcement usually
        resolves surface-level violations.

        Only violation counts are recorded in context, never the constraint
        terms themselves (which may describe sensitive user preferences).
        """
        if not constraints:
            return None
        normalised = content.lower()
        violated_count = sum(
            1 for c in constraints if c.lower() not in normalised
        )
        if violated_count:
            return PipelineError(
                error_code=CONSTRAINT_VIOLATED,
                severity=ErrorSeverity.RETRYABLE,
                agent_id=self.agent_id,
                run_id=run_id,
                topic_id=topic_id,
                message=(
                    f"{violated_count} of {len(constraints)} constraint(s) "
                    "not satisfied in output"
                ),
                retry_hint=(
                    "Re-run with stronger constraint enforcement in the system prompt"
                ),
                context={
                    "topic_id": topic_id,
                    "total_constraints": len(constraints),
                    "violated_count": violated_count,
                },
            )
        return None
