"""
Phase 1 orchestrator — agent call layer.

Switches between local (in-process) and HTTP agent calls based on
USE_LOCAL_AGENTS (default: true for local dev).

Local mode: imports each agent's run_agent() directly.
HTTP mode:  calls agent endpoints with AAP token via httpx.
           Not yet implemented — see [A2A-TODO] below.

[A2A-TODO] HTTP production path:
  - GET agent card from registry by capability
  - Verify card signature (once registry signing is active)
  - POST to card.endpoint/run with:
      Authorization: X-MAS-Secret  (current; swap for DPoP once DPOP-TODO is done)
      body: input.model_dump_json()
  - Deserialise response into the appropriate Output schema
  See ARCHITECTURE.md §4 for the A2A protocol design.
"""
from __future__ import annotations

import os
from typing import Any

from src.agents.filter_agent.agent import run_agent as _filter_run
from src.agents.heat_scorer.agent import run_agent as _heat_run
from src.agents.phase1_judge.agent import run_agent as _judge_run
from src.agents.search_worker.agent import run_agent as _search_run
from src.agents.selector.agent import run_agent as _selector_run
from src.common.schemas import (
    FilterAgentInput,
    FilterAgentOutput,
    HeatScorerInput,
    HeatScorerOutput,
    Phase1JudgeInput,
    Phase1JudgeOutput,
    SearchWorkerInput,
    SearchWorkerOutput,
    SelectorInput,
    SelectorOutput,
)


def _use_local() -> bool:
    return os.getenv("USE_LOCAL_AGENTS", "true").lower() not in ("false", "0", "no")


async def call_search_worker(inp: SearchWorkerInput) -> SearchWorkerOutput:
    if _use_local():
        return await _search_run(inp)
    raise NotImplementedError(
        "HTTP agent calls not yet implemented. "
        "Set USE_LOCAL_AGENTS=true or implement [A2A-TODO] HTTP path."
    )


async def call_heat_scorer(
    inp: HeatScorerInput,
    *,
    model_id: str | None = None,
) -> HeatScorerOutput:
    if _use_local():
        return await _heat_run(inp, model_id=model_id)
    raise NotImplementedError("HTTP agent calls not yet implemented.")


async def call_filter_agent(
    inp: FilterAgentInput,
    *,
    model_id: str | None = None,
) -> FilterAgentOutput:
    if _use_local():
        return await _filter_run(inp, model_id=model_id)
    raise NotImplementedError("HTTP agent calls not yet implemented.")


async def call_selector(
    inp: SelectorInput,
    *,
    model_id: str | None = None,
) -> SelectorOutput:
    if _use_local():
        return await _selector_run(inp, model_id=model_id)
    raise NotImplementedError("HTTP agent calls not yet implemented.")


async def call_phase1_judge(
    inp: Phase1JudgeInput,
    *,
    model_id: str | None = None,
) -> Phase1JudgeOutput:
    if _use_local():
        return await _judge_run(inp, model_id=model_id)
    raise NotImplementedError("HTTP agent calls not yet implemented.")
