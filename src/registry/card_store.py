"""
In-memory config store for the agent registry.

Backed by a plain dict — the interface (save/get/update/list) is identical to
what a FernetStorage-backed implementation would expose, so this can be swapped
without touching callers.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.registry.models import AgentConfig

_store: dict[str, AgentConfig] = {}


def save_agent_config(config: AgentConfig) -> None:
    _store[config.agent_id] = config


def get_agent_config(agent_id: str) -> AgentConfig:
    config = _store.get(agent_id)
    if config is None:
        raise KeyError(
            f"No config found for agent '{agent_id}'. "
            "Ensure the registry has been seeded with default configs."
        )
    return config


def update_agent_config(agent_id: str, updates: dict, updated_by: str) -> AgentConfig:
    config = get_agent_config(agent_id)
    updated = config.model_copy(
        update={
            **updates,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": updated_by,
        }
    )
    _store[agent_id] = updated
    return updated


def list_agent_configs() -> list[AgentConfig]:
    return list(_store.values())
