"""
Pipeline error protocol.

All agent business logic that can fail should return AgentResult rather than
raising. AgentResult.success=False with a populated PipelineError lets the
orchestrator route the failure cleanly (retry, skip, degrade, abort) without
try/except scattered through graph nodes.

PipelineException is available for the rare cases where raising is cleaner
than returning (e.g. inside a deeply nested sync helper where the call stack
does not thread AgentResult up naturally).

PHI safety rules:
  - PipelineError.message must never contain article content, summaries, or
    any user-originated text — describe the structural problem, not the data.
  - PipelineError.context is for metadata (counts, lengths, IDs, thresholds).
    Never put content values in context. The DataQualityValidator enforces this;
    callers must respect it too.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorSeverity(str, Enum):
    RETRYABLE = "RETRYABLE"  # orchestrator should retry the agent call
    SKIP = "SKIP"            # skip this item and continue with the rest
    FATAL = "FATAL"          # abort the run; pipeline cannot proceed
    DEGRADED = "DEGRADED"    # continue with reduced quality / partial output


class ResultConfidence(str, Enum):
    FULL = "full"          # fetched full content
    SNIPPET = "snippet"    # fetch failed, snippet only
    PARTIAL = "partial"    # fetched but truncated at TOKEN_BUDGET
    INJECTED = "injected"  # injection detected, content discarded
    SCRUBBED = "scrubbed"  # PHI removed, content altered


class PipelineError(BaseModel):
    """
    Structured error record threaded through the pipeline.

    context holds only metadata (counts, lengths, IDs, thresholds).
    It must never contain raw article content, summaries, or PHI.
    """
    error_code: str
    severity: ErrorSeverity
    agent_id: str
    run_id: str
    topic_id: str = ""
    message: str
    retry_hint: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """
    Universal agent return envelope.

    Typed agent outputs inherit from this and add their specific fields.
    When used as a standalone envelope (not subclassed), populate `data`
    with the output value.

    Orchestrator routing on failure:
      RETRYABLE  → retry the agent, up to MAX_RETRIES
      SKIP       → skip this item; continue with remaining items
      DEGRADED   → continue; flag run as partial in RunState
      FATAL      → mark run as failed; do not process further items
    """
    success: bool = True
    data: Any = None
    error: PipelineError | None = None
    warnings: list[str] = Field(default_factory=list)


class PipelineException(Exception):
    """
    Wraps a PipelineError for cases where raising is cleaner than returning
    AgentResult (e.g. deep sync helpers that don't thread AgentResult up).

    Catch at the agent boundary and convert to AgentResult(success=False,
    error=exc.error) before returning to the orchestrator.
    """

    def __init__(self, error: PipelineError) -> None:
        self.error = error
        super().__init__(error.message)
