"""
Tests for resolve_model() in src/common/agent_bootstrap.py.

All tests are pure unit tests — no network, no Ollama, no Anthropic API.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

import src.common.agent_bootstrap as ab_module
from src.common.agent_bootstrap import resolve_model


# ── No override active ────────────────────────────────────────────────────────

def test_returns_registry_values_when_env_unset():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MODEL_OVERRIDE", None)
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "heat-scorer")
    assert provider == "ollama"
    assert model_id == "gemma4:e4b"


def test_returns_registry_values_when_override_empty():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": ""}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "heat-scorer")
    assert provider == "ollama"
    assert model_id == "gemma4:e4b"


def test_returns_registry_values_when_override_is_comment():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "#claude-haiku"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "selector")
    assert provider == "ollama"
    assert model_id == "gemma4:e4b"


# ── Mapped override values ─────────────────────────────────────────────────────

def test_claude_haiku_full_id_maps_to_anthropic():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku-4-5-20251001"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "heat-scorer")
    assert provider == "anthropic"
    assert model_id == "claude-haiku-4-5-20251001"


def test_claude_haiku_shorthand_maps_to_full_id():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "filter-agent")
    assert provider == "anthropic"
    assert model_id == "claude-haiku-4-5-20251001"


def test_claude_sonnet_shorthand_maps_correctly():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-sonnet"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "selector")
    assert provider == "anthropic"
    assert model_id == "claude-sonnet-4-6"


def test_gemma4_override_stays_ollama():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "gemma4:e4b"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "phase1-judge")
    assert provider == "ollama"
    assert model_id == "gemma4:e4b"


# ── Unknown override — provider inferred ──────────────────────────────────────

def test_unknown_claude_prefix_infers_anthropic():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-opus-4-7"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "heat-scorer")
    assert provider == "anthropic"
    assert model_id == "claude-opus-4-7"


def test_unknown_non_claude_string_infers_ollama():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "llama3.2:latest"}):
        provider, model_id = resolve_model("gemma4:e4b", "ollama", "heat-scorer")
    assert provider == "ollama"
    assert model_id == "llama3.2:latest"


# ── Exempt agents ──────────────────────────────────────────────────────────────

def test_summarizer_exempt_from_override():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku-4-5-20251001"}):
        provider, model_id = resolve_model("claude-sonnet-4-6", "anthropic", "summarizer")
    assert provider == "anthropic"
    assert model_id == "claude-sonnet-4-6"


def test_reviewer_exempt_from_override():
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku-4-5-20251001"}):
        provider, model_id = resolve_model("claude-sonnet-4-6", "anthropic", "reviewer")
    assert provider == "anthropic"
    assert model_id == "claude-sonnet-4-6"


def test_exempt_agent_ignores_override_completely():
    """Even a gemma override is ignored for exempt agents."""
    with patch.dict(os.environ, {"MODEL_OVERRIDE": "gemma4:e4b"}):
        provider, model_id = resolve_model("claude-sonnet-4-6", "anthropic", "summarizer")
    assert provider == "anthropic"
    assert model_id == "claude-sonnet-4-6"


# ── Warning logged when override active ───────────────────────────────────────

def test_warning_logged_when_override_active():
    mock_logger = MagicMock()
    with patch.object(ab_module, "logger", mock_logger):
        with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku-4-5-20251001"}):
            resolve_model("gemma4:e4b", "ollama", "heat-scorer")

    mock_logger.warning.assert_called_once()
    call_args, call_kwargs = mock_logger.warning.call_args
    assert call_args[0] == "model.override.active"
    extra = call_kwargs["extra"]
    assert extra["agent_id"] == "heat-scorer"
    assert extra["override_model"] == "claude-haiku-4-5-20251001"
    assert extra["override_provider"] == "anthropic"
    assert extra["registry_model"] == "gemma4:e4b"


def test_no_warning_when_no_override():
    mock_logger = MagicMock()
    with patch.object(ab_module, "logger", mock_logger):
        with patch.dict(os.environ, {"MODEL_OVERRIDE": ""}):
            resolve_model("gemma4:e4b", "ollama", "heat-scorer")

    mock_logger.warning.assert_not_called()


def test_no_warning_for_exempt_agent_even_with_override():
    mock_logger = MagicMock()
    with patch.object(ab_module, "logger", mock_logger):
        with patch.dict(os.environ, {"MODEL_OVERRIDE": "claude-haiku-4-5-20251001"}):
            resolve_model("claude-sonnet-4-6", "anthropic", "summarizer")

    mock_logger.warning.assert_not_called()
