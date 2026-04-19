from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Authoritative configuration record for one agent.

    Stored in the registry and fetched at agent startup via bootstrap_agent().
    Agents must not hardcode any of these fields — the registry is the single
    source of truth for model assignment, capabilities, and network placement.
    """
    agent_id: str
    model_provider: str = Field(
        description="anthropic | ollama | tavily | none"
    )
    model_id: str = Field(
        description="Model identifier, e.g. 'claude-sonnet-4-6', 'gemma4:e4b', or 'none'"
    )
    capabilities: list[str] = Field(
        description="Operations this agent is permitted to perform"
    )
    token_budget: int = Field(
        ge=0,
        description="Max tokens per run; 0 for deterministic agents that make no LLM calls",
    )
    prompt_version: str = Field(
        description="Prompt file version to load, e.g. 'v1.0'"
    )
    allowed_callers: list[str] = Field(
        description="Agent IDs permitted to call this agent's /run endpoint"
    )
    network_tier: str = Field(
        description="compute | orchestration | identity — maps to Docker/Azure subnet"
    )
    timeout_seconds: int = Field(ge=1, le=300)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_by: str = "system"
