"""
Tests for agent bootstrap and registry config endpoints.

Uses httpx.AsyncClient with ASGITransport to call the registry FastAPI app
directly — no network required, no running server.
"""
from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
import src.registry.registry_server as registry_module

from src.auth.token_service import mint_token
from src.auth.workload_identity import LocalDevIdentityProvider
from src.common.agent_bootstrap import bootstrap_agent
from src.registry import card_store
from src.registry.models import AgentConfig
from src.registry.registry_server import app as registry_app

# Shared test secret — patched into registry_module._SECRET_KEY before each test.
_TEST_SECRET = "test-secret-bootstrap"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_aap_token(capabilities: list[str]) -> str:
    return mint_token(
        subject="test-caller",
        agent_name="test-caller",
        model_provider="none",
        model_id="none",
        capabilities=capabilities,
        task_id=str(uuid.uuid4()),
        task_description="test",
        data_sensitivity="internal",
        oversight_required=False,
        oversight_level="none",
        context={},
        secret_key=_TEST_SECRET,
    )


def _transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=registry_app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_registry_secret(monkeypatch):
    """Patch the registry's module-level _SECRET_KEY for every test."""
    monkeypatch.setattr(registry_module, "_SECRET_KEY", _TEST_SECRET)


@pytest.fixture(autouse=True)
def isolate_card_store():
    """Reset the in-memory store before and after every test."""
    card_store._store.clear()
    yield
    card_store._store.clear()


@pytest.fixture()
def identity_provider():
    return LocalDevIdentityProvider(
        secret_key=_TEST_SECRET,
        agent_id="search-worker",
    )


@pytest.fixture()
def seeded_store():
    """Seed a known config for search-worker."""
    config = AgentConfig(
        agent_id="search-worker",
        model_provider="tavily",
        model_id="none",
        capabilities=["search.web"],
        token_budget=0,
        prompt_version="v1.0",
        allowed_callers=["orchestrator-phase1"],
        network_tier="compute",
        timeout_seconds=30,
    )
    card_store.save_agent_config(config)
    return config


@pytest_asyncio.fixture()
async def registry_client():
    """
    AsyncClient pointing at the registry app with auth headers pre-set.
    Triggers startup seeding so the 8 default agent configs are present.
    """
    async with httpx.AsyncClient(
        transport=_transport(),
        base_url="http://test",
        headers={"X-MAS-Secret": _TEST_SECRET},
    ) as client:
        from src.registry.registry_server import startup
        await startup()
        yield client


# ── Test 1: bootstrap_agent() returns correct AgentConfig ─────────────────────

@pytest.mark.asyncio
async def test_bootstrap_returns_correct_config(monkeypatch, seeded_store, identity_provider):
    monkeypatch.setenv("MAS_SECRET_KEY", _TEST_SECRET)

    async with httpx.AsyncClient(
        transport=_transport(), base_url="http://test"
    ) as client:
        config = await bootstrap_agent(
            registry_url="http://test",
            identity_provider=identity_provider,
            agent_id="search-worker",
            _client=client,
        )

    assert config.agent_id == "search-worker"
    assert config.model_provider == "tavily"
    assert config.model_id == "none"
    assert "search.web" in config.capabilities
    assert config.network_tier == "compute"
    assert config.token_budget == 0


# ── Test 2: Agent self-configures from registry response ──────────────────────

@pytest.mark.asyncio
async def test_agent_self_configures_from_registry(monkeypatch, seeded_store, identity_provider):
    """
    bootstrap_agent() returns an AgentConfig with all fields needed for
    self-configuration. Verify model, capabilities, budget, version, tier.
    """
    monkeypatch.setenv("MAS_SECRET_KEY", _TEST_SECRET)

    async with httpx.AsyncClient(
        transport=_transport(), base_url="http://test"
    ) as client:
        config = await bootstrap_agent(
            registry_url="http://test",
            identity_provider=identity_provider,
            agent_id="search-worker",
            _client=client,
        )

    assert config.model_provider == "tavily"
    assert config.model_id == "none"
    assert config.capabilities == ["search.web"]
    assert config.token_budget == 0
    assert config.prompt_version == "v1.0"
    assert config.network_tier == "compute"
    assert config.timeout_seconds == 30
    assert config.allowed_callers == ["orchestrator-phase1"]


# ── Test 3: Missing agent config raises clear error ────────────────────────────

@pytest.mark.asyncio
async def test_missing_agent_config_raises_error(monkeypatch, identity_provider):
    monkeypatch.setenv("MAS_SECRET_KEY", _TEST_SECRET)
    # Store is empty (isolate_card_store clears it before each test).

    async with httpx.AsyncClient(
        transport=_transport(), base_url="http://test"
    ) as client:
        with pytest.raises(ValueError, match="Registry has no config for agent 'ghost-agent'"):
            await bootstrap_agent(
                registry_url="http://test",
                identity_provider=identity_provider,
                agent_id="ghost-agent",
                _client=client,
            )


# ── Test 4: Config update is audited with updated_by ──────────────────────────

@pytest.mark.asyncio
async def test_config_update_is_audited(registry_client):
    admin_token = _make_aap_token(["registry.config.write"])

    response = await registry_client.put(
        "/agents/search-worker/config",
        json={"timeout_seconds": 45},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["timeout_seconds"] == 45
    assert data["updated_by"] == "test-caller"
    assert data["updated_at"] is not None

    stored = card_store.get_agent_config("search-worker")
    assert stored.timeout_seconds == 45
    assert stored.updated_by == "test-caller"


# ── Test 5: Worker agent cannot call PUT /config ──────────────────────────────

@pytest.mark.asyncio
async def test_worker_cannot_put_config(registry_client):
    """
    A worker-tier AAP token (capabilities: ["search.web"]) must be rejected
    with 403 on PUT /config. The capability check is defence-in-depth;
    the primary enforcement is network isolation (compute subnet cannot
    reach the registry PUT endpoint at the network layer).
    """
    worker_token = _make_aap_token(["search.web"])

    response = await registry_client.put(
        "/agents/search-worker/config",
        json={"timeout_seconds": 99},
        headers={"Authorization": f"Bearer {worker_token}"},
    )
    assert response.status_code == 403
    assert "registry.config.write" in response.json()["detail"]


@pytest.mark.asyncio
async def test_worker_cannot_put_config_without_token(registry_client):
    """PUT without any Authorization header must also be rejected with 403."""
    response = await registry_client.put(
        "/agents/search-worker/config",
        json={"timeout_seconds": 99},
    )
    assert response.status_code == 403


# ── Test 6: Registry seeds all 8 default configs on startup ───────────────────

@pytest.mark.asyncio
async def test_registry_seeds_all_agents(registry_client):
    response = await registry_client.get("/config/defaults")
    assert response.status_code == 200
    agent_ids = {d["agent_id"] for d in response.json()}
    expected = {
        "search-worker", "heat-scorer", "filter-agent", "selector",
        "phase1-judge", "summarizer", "reviewer", "relevance-gate",
    }
    assert expected == agent_ids


@pytest.mark.asyncio
async def test_correct_model_assignments(registry_client):
    """Verify model assignments match the architecture spec (ARCHITECTURE.md §6)."""
    response = await registry_client.get("/config/defaults")
    configs = {d["agent_id"]: d for d in response.json()}

    assert configs["search-worker"]["model_provider"] == "tavily"
    assert configs["heat-scorer"]["model_provider"] == "ollama"
    assert configs["heat-scorer"]["model_id"] == "gemma4:e4b"
    assert configs["filter-agent"]["model_provider"] == "none"
    assert configs["selector"]["model_provider"] == "none"
    assert configs["phase1-judge"]["model_provider"] == "ollama"
    assert configs["phase1-judge"]["model_id"] == "gemma4:e4b"
    assert configs["summarizer"]["model_provider"] == "anthropic"
    assert configs["summarizer"]["model_id"] == "claude-sonnet-4-6"
    assert configs["reviewer"]["model_provider"] == "anthropic"
    assert configs["reviewer"]["model_id"] == "claude-sonnet-4-6"
    assert configs["relevance-gate"]["model_provider"] == "ollama"
    assert configs["relevance-gate"]["model_id"] == "gemma4:e4b"


@pytest.mark.asyncio
async def test_get_agent_config_requires_auth():
    """GET /agents/{id}/config must reject requests without X-MAS-Secret."""
    async with httpx.AsyncClient(
        transport=_transport(), base_url="http://test"
    ) as client:
        from src.registry.registry_server import startup
        await startup()
        response = await client.get("/agents/search-worker/config")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_nonexistent_agent_config_returns_404(registry_client):
    response = await registry_client.get("/agents/does-not-exist/config")
    assert response.status_code == 404
