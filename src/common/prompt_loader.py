from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_ROOT = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(agent_name: str, version: str = "v1.0") -> dict[str, Any]:
    """
    Load a prompt YAML for *agent_name* at *version* and return it tagged
    with metadata so every downstream call is traceable.

    Directory convention: prompts/<agent_name>/<version>.yaml
    """
    path = _PROMPTS_ROOT / agent_name / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    data["_meta"] = {
        "agent": agent_name,
        "version": version,
        "path": str(path),
    }
    return data


def get_system_prompt(agent_name: str, version: str = "v1.0") -> str:
    """Return the assembled system prompt (preamble + system) for an agent."""
    prompt = load_prompt(agent_name, version)
    parts: list[str] = []
    if preamble := prompt.get("injection_defense_preamble"):
        parts.append(preamble.strip())
    if system := prompt.get("system"):
        parts.append(system.strip())
    return "\n\n".join(parts)


def render_user_prompt(
    agent_name: str,
    variables: dict[str, Any],
    version: str = "v1.0",
) -> str:
    """Render the user_template with *variables* substituted."""
    prompt = load_prompt(agent_name, version)
    template: str = prompt.get("user_template", "")
    return template.format(**variables)


def make_run_config(
    agent_name: str,
    version: str = "v1.0",
    *,
    run_id: str | None = None,
    topic_id: str | None = None,
) -> dict[str, Any]:
    """
    Return a LangChain-compatible run config dict for LangSmith tracing.

    Pass the returned dict as ``config=`` on any LangChain LLM call:

        response = await llm.ainvoke(messages, config=make_run_config(...))

    For direct Ollama / Anthropic SDK calls, pass ``metadata`` and ``tags``
    separately via the SDK's callback or metadata arguments.
    """
    tags: list[str] = [f"agent:{agent_name}", f"prompt:{version}"]
    metadata: dict[str, Any] = {
        "prompt_version": version,
        "agent_id": agent_name,
    }
    if run_id:
        tags.append(f"run:{run_id}")
        metadata["run_id"] = run_id
    if topic_id:
        tags.append(f"topic:{topic_id}")
        metadata["topic_id"] = topic_id

    return {
        "run_name": f"{agent_name}:{version}",
        "tags": tags,
        "metadata": metadata,
    }
