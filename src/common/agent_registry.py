"""In-memory agent registry. Populated at each agent's startup."""
from __future__ import annotations

from typing import Any

_registry: dict[str, dict[str, Any]] = {}


def register(agent_id: str, *, capabilities: list[str], endpoint: str) -> None:
    _registry[agent_id] = {
        "agent_id": agent_id,
        "capabilities": capabilities,
        "endpoint": endpoint,
    }


def get(agent_id: str) -> dict[str, Any] | None:
    return _registry.get(agent_id)


def list_agents() -> list[dict[str, Any]]:
    return list(_registry.values())
