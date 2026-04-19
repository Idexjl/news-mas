"""Tests for configure_langsmith() in src/common/observability.py."""
from __future__ import annotations

import os
import logging

import pytest


def _clear_langchain_env() -> None:
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"):
        os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Each test starts with a clean slate of LANGCHAIN_* vars."""
    _clear_langchain_env()
    yield
    _clear_langchain_env()


def test_configure_langsmith_sets_env_vars(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")

    from src.common.observability import configure_langsmith
    result = configure_langsmith()

    assert result is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "test-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
    assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"


def test_configure_langsmith_missing_key_returns_false_and_warns(monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    from src.common.observability import configure_langsmith
    result = configure_langsmith()

    assert result is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") != "true"
    # get_logger() uses propagate=False with its own StreamHandler, so warnings
    # reach stderr rather than caplog. Check stderr directly.
    assert "LangSmith" in capsys.readouterr().err


def test_configure_langsmith_defaults_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    from src.common.observability import configure_langsmith
    configure_langsmith()

    assert os.environ["LANGCHAIN_PROJECT"] == "news-mas-dev"


def test_configure_langsmith_idempotent(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    from src.common.observability import configure_langsmith
    first = configure_langsmith()
    second = configure_langsmith()

    assert first is True
    assert second is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_configure_langsmith_langchain_api_key_fallback(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "fallback-key")

    from src.common.observability import configure_langsmith
    result = configure_langsmith()

    assert result is True
    assert os.environ["LANGCHAIN_API_KEY"] == "fallback-key"
