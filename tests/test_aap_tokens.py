"""
TDD spec for src.auth.token_service — Agent Authorization Protocol (AAP) tokens.

All tests are written BEFORE the implementation exists. They fail with
ImportError until src/auth/token_service.py is created. Once the
implementation lands, run this file as the regression suite.

    pytest tests/test_aap_tokens.py -v

No external JWT library is required in this file — helpers use stdlib
(base64 + hmac + json) to build fixture tokens, so the only import that
will fail is the one from the not-yet-existing implementation module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from src.auth.token_service import MAX_DEPTH, exchange_token, mint_token, validate_token


# ── helpers (stdlib-only JWT encoding / decoding for test fixtures) ────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


def _decode_payload(token: str) -> dict[str, Any]:
    """Decode the payload segment without signature verification."""
    _, payload_b64, _ = token.split(".")
    return json.loads(_b64url_decode(payload_b64))


def _encode_hs256(payload: dict[str, Any], secret: str) -> str:
    """Encode a minimal HS256 JWT using stdlib only — for test fixtures."""
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(sig)}"


def _make_full_aap_payload(
    *,
    sub: str = "agent:test",
    iat: int | None = None,
    exp: int | None = None,
    omit_sections: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    Return a payload dict with all seven AAP sections.
    Pass omit_sections to drop specific sections — used to test missing-claim
    validation without calling mint_token.
    """
    now = iat if iat is not None else int(time.time())
    payload: dict[str, Any] = {
        "sub": sub,
        "iat": now,
        "exp": exp if exp is not None else now + 3600,
        "aap_agent": {
            "agent_name": "test_agent",
            "model_provider": "anthropic",
            "model_id": "claude-haiku-4-5",
        },
        "aap_task": {
            "task_id": "task-fixture",
            "task_description": "Fixture task",
            "data_sensitivity": "internal",
        },
        "aap_capabilities": ["read"],
        "aap_oversight": {"oversight_required": False, "oversight_level": "none"},
        "aap_delegation": {"depth": 0, "chain": [{"agent_name": "test_agent"}], "max_depth": 3},
        "aap_context": {},
        "aap_audit": {"token_id": str(uuid.uuid4()), "minted_at": now},
    }
    for section in omit_sections:
        payload.pop(section, None)
    return payload


# ── shared constants & baseline kwargs ────────────────────────────────────────

_SECRET = "test-secret-not-for-production"

# Minimal valid kwargs for mint_token — used as the baseline across all tests.
_MINT_KWARGS: dict[str, Any] = dict(
    subject="agent:search-worker",
    agent_name="search_worker",
    model_provider="anthropic",
    model_id="claude-sonnet-4-6",
    capabilities=["web_search", "content_fetch"],
    task_id="task-abc-001",
    task_description="Fetch top headlines for clustering",
    data_sensitivity="internal",
    oversight_required=False,
    oversight_level="none",
    context={"run_id": "run-xyz", "topic": "AI regulation"},
    secret_key=_SECRET,
)

# Minimal kwargs for the receiving agent in exchange_token calls.
_EXCHANGE_KWARGS: dict[str, Any] = dict(
    new_agent_name="heat_scorer",
    new_model_provider="anthropic",
    new_model_id="claude-haiku-4-5",
    new_capabilities=["score_article", "classify_content"],
    secret_key=_SECRET,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def token() -> str:
    return mint_token(**_MINT_KWARGS)


@pytest.fixture()
def claims(token: str) -> dict[str, Any]:
    return validate_token(token, secret_key=_SECRET)


@pytest.fixture()
def exchanged(token: str) -> str:
    return exchange_token(token=token, **_EXCHANGE_KWARGS)


@pytest.fixture()
def exchanged_claims(exchanged: str) -> dict[str, Any]:
    return validate_token(exchanged, secret_key=_SECRET)


# ── 1. Token minting ───────────────────────────────────────────────────────────

def test_mint_returns_string(token: str) -> None:
    assert isinstance(token, str)


def test_mint_produces_three_part_jwt(token: str) -> None:
    # Header.Payload.Signature — three segments joined by exactly two dots.
    assert token.count(".") == 2


def test_all_seven_aap_sections_present(claims: dict) -> None:
    required = {
        "aap_agent",
        "aap_task",
        "aap_capabilities",
        "aap_oversight",
        "aap_delegation",
        "aap_context",
        "aap_audit",
    }
    missing = required - claims.keys()
    assert not missing, f"Missing AAP sections: {missing}"


def test_aap_agent_contains_agent_name(claims: dict) -> None:
    assert claims["aap_agent"]["agent_name"] == "search_worker"


def test_aap_agent_contains_model_provider(claims: dict) -> None:
    assert claims["aap_agent"]["model_provider"] == "anthropic"


def test_aap_agent_contains_model_id(claims: dict) -> None:
    assert claims["aap_agent"]["model_id"] == "claude-sonnet-4-6"


def test_aap_task_contains_task_id(claims: dict) -> None:
    assert claims["aap_task"]["task_id"] == "task-abc-001"


def test_aap_task_contains_task_description(claims: dict) -> None:
    assert claims["aap_task"]["task_description"] == "Fetch top headlines for clustering"


def test_aap_task_contains_data_sensitivity(claims: dict) -> None:
    assert claims["aap_task"]["data_sensitivity"] == "internal"


def test_aap_capabilities_is_list(claims: dict) -> None:
    assert isinstance(claims["aap_capabilities"], list)


def test_aap_capabilities_all_strings(claims: dict) -> None:
    assert all(isinstance(c, str) for c in claims["aap_capabilities"])


def test_aap_capabilities_matches_declared_value(claims: dict) -> None:
    assert set(claims["aap_capabilities"]) == {"web_search", "content_fetch"}


def test_aap_oversight_required_field(claims: dict) -> None:
    assert claims["aap_oversight"]["oversight_required"] is False


def test_aap_oversight_level_field(claims: dict) -> None:
    assert claims["aap_oversight"]["oversight_level"] == "none"


def test_aap_delegation_starts_at_depth_zero(claims: dict) -> None:
    assert claims["aap_delegation"]["depth"] == 0


def test_aap_delegation_initial_chain_has_one_entry(claims: dict) -> None:
    chain = claims["aap_delegation"]["chain"]
    assert isinstance(chain, list)
    assert len(chain) == 1


def test_aap_delegation_initial_chain_entry_is_minting_agent(claims: dict) -> None:
    assert claims["aap_delegation"]["chain"][0]["agent_name"] == "search_worker"


def test_aap_delegation_stores_max_depth(claims: dict) -> None:
    # max_depth must be embedded in the token so exchange_token can enforce it
    # without extra parameters.
    assert "max_depth" in claims["aap_delegation"]


def test_aap_context_carries_context_dict(claims: dict) -> None:
    assert claims["aap_context"]["run_id"] == "run-xyz"
    assert claims["aap_context"]["topic"] == "AI regulation"


def test_aap_audit_has_token_id(claims: dict) -> None:
    token_id = claims["aap_audit"]["token_id"]
    assert isinstance(token_id, str)
    uuid.UUID(token_id)  # raises ValueError if not a valid UUID


def test_aap_audit_has_minted_at_timestamp(claims: dict) -> None:
    minted_at = claims["aap_audit"]["minted_at"]
    assert isinstance(minted_at, int)
    assert minted_at <= int(time.time())


def test_standard_sub_claim(claims: dict) -> None:
    assert claims["sub"] == "agent:search-worker"


def test_standard_exp_claim_is_in_future(claims: dict) -> None:
    assert claims["exp"] > int(time.time())


def test_standard_iat_claim_is_not_in_future(claims: dict) -> None:
    assert claims["iat"] <= int(time.time())


# ── 2. Token exchange ──────────────────────────────────────────────────────────

def test_exchange_returns_string(exchanged: str) -> None:
    assert isinstance(exchanged, str)


def test_exchange_produces_valid_jwt(exchanged: str) -> None:
    assert exchanged.count(".") == 2


def test_exchange_preserves_sub(claims: dict, exchanged_claims: dict) -> None:
    assert exchanged_claims["sub"] == claims["sub"]


def test_exchange_preserves_aap_task(claims: dict, exchanged_claims: dict) -> None:
    # Task context (including data_sensitivity) must survive delegation unchanged.
    assert exchanged_claims["aap_task"] == claims["aap_task"]


def test_exchange_preserves_aap_oversight(claims: dict, exchanged_claims: dict) -> None:
    # Oversight requirements bind to the task, not the agent — must not change.
    assert exchanged_claims["aap_oversight"] == claims["aap_oversight"]


def test_exchange_increments_delegation_depth(claims: dict, exchanged_claims: dict) -> None:
    assert exchanged_claims["aap_delegation"]["depth"] == claims["aap_delegation"]["depth"] + 1


def test_exchange_appends_new_agent_to_delegation_chain(exchanged_claims: dict) -> None:
    chain = exchanged_claims["aap_delegation"]["chain"]
    assert len(chain) == 2
    assert chain[-1]["agent_name"] == "heat_scorer"


def test_exchange_chain_retains_original_agent_at_index_zero(
    claims: dict, exchanged_claims: dict
) -> None:
    assert exchanged_claims["aap_delegation"]["chain"][0] == claims["aap_delegation"]["chain"][0]


def test_exchange_updates_aap_agent_name(exchanged_claims: dict) -> None:
    assert exchanged_claims["aap_agent"]["agent_name"] == "heat_scorer"


def test_exchange_updates_aap_agent_model_provider(exchanged_claims: dict) -> None:
    assert exchanged_claims["aap_agent"]["model_provider"] == "anthropic"


def test_exchange_updates_aap_agent_model_id(exchanged_claims: dict) -> None:
    assert exchanged_claims["aap_agent"]["model_id"] == "claude-haiku-4-5"


def test_exchange_at_max_depth_raises_value_error() -> None:
    # max_depth=1 → one exchange allowed (depth 0→1), second must raise.
    t = mint_token(**{**_MINT_KWARGS, "max_depth": 1})
    t = exchange_token(token=t, **_EXCHANGE_KWARGS)  # depth 0 → 1, OK
    with pytest.raises(ValueError, match=r"(?i)max.{0,15}depth|delegation.{0,15}limit|depth.{0,15}exceed"):
        exchange_token(token=t, **_EXCHANGE_KWARGS)  # depth would exceed max_depth=1


def test_exchange_max_depth_two_allows_two_exchanges() -> None:
    # Positive case: max_depth=2 means exactly 2 exchanges are valid.
    t = mint_token(**{**_MINT_KWARGS, "max_depth": 2})
    t = exchange_token(token=t, **_EXCHANGE_KWARGS)  # depth → 1
    t = exchange_token(token=t, **_EXCHANGE_KWARGS)  # depth → 2
    # No exception raised — both exchanges succeeded.
    result = validate_token(t, secret_key=_SECRET)
    assert result["aap_delegation"]["depth"] == 2


def test_exchange_at_max_depth_two_third_raises() -> None:
    t = mint_token(**{**_MINT_KWARGS, "max_depth": 2})
    t = exchange_token(token=t, **_EXCHANGE_KWARGS)
    t = exchange_token(token=t, **_EXCHANGE_KWARGS)
    with pytest.raises(ValueError):
        exchange_token(token=t, **_EXCHANGE_KWARGS)  # would be depth 3 > max_depth=2


# ── 3. Token validation ────────────────────────────────────────────────────────

def test_validate_returns_dict_for_valid_token(token: str) -> None:
    result = validate_token(token, secret_key=_SECRET)
    assert isinstance(result, dict)


def test_validate_returned_dict_contains_sub(token: str) -> None:
    result = validate_token(token, secret_key=_SECRET)
    assert "sub" in result


def test_validate_returned_dict_contains_all_aap_sections(token: str) -> None:
    result = validate_token(token, secret_key=_SECRET)
    for section in ("aap_agent", "aap_task", "aap_capabilities",
                    "aap_oversight", "aap_delegation", "aap_context", "aap_audit"):
        assert section in result, f"validate_token result missing '{section}'"


def test_validate_expired_token_raises_value_error() -> None:
    expired = _encode_hs256(
        _make_full_aap_payload(iat=int(time.time()) - 120, exp=int(time.time()) - 60),
        _SECRET,
    )
    with pytest.raises(ValueError, match=r"(?i)expir"):
        validate_token(expired, secret_key=_SECRET)


def test_validate_wrong_secret_raises_value_error(token: str) -> None:
    with pytest.raises(ValueError, match=r"(?i)signature|invalid|secret|verification"):
        validate_token(token, secret_key="completely-wrong-secret")


def test_validate_tampered_signature_raises_value_error(token: str) -> None:
    # Corrupt the last character of the signature segment.
    rest, sig = token.rsplit(".", 1)
    bad_sig = sig[:-1] + ("A" if sig[-1] != "A" else "B")
    with pytest.raises(ValueError):
        validate_token(f"{rest}.{bad_sig}", secret_key=_SECRET)


def test_validate_missing_all_aap_claims_raises_value_error() -> None:
    # A JWT with the correct secret but no AAP sections must be rejected.
    bare = _encode_hs256(
        {"sub": "agent:test", "iat": int(time.time()), "exp": int(time.time()) + 3600},
        _SECRET,
    )
    with pytest.raises(ValueError, match=r"(?i)missing|required|aap"):
        validate_token(bare, secret_key=_SECRET)


@pytest.mark.parametrize("missing_section", [
    "aap_agent",
    "aap_task",
    "aap_capabilities",
    "aap_oversight",
    "aap_delegation",
    "aap_context",
    "aap_audit",
])
def test_validate_missing_single_aap_section_raises_value_error(missing_section: str) -> None:
    """Removing any one of the seven AAP sections must be detected and rejected."""
    payload = _make_full_aap_payload(omit_sections=(missing_section,))
    partial_token = _encode_hs256(payload, _SECRET)
    with pytest.raises(ValueError, match=r"(?i)missing|required|aap"):
        validate_token(partial_token, secret_key=_SECRET)


# ── 4. Capability constraints ──────────────────────────────────────────────────

def test_capabilities_match_minted_value(claims: dict) -> None:
    assert set(claims["aap_capabilities"]) == {"web_search", "content_fetch"}


def test_exchange_replaces_capabilities_with_new_agent_capabilities(
    exchanged_claims: dict,
) -> None:
    assert set(exchanged_claims["aap_capabilities"]) == {"score_article", "classify_content"}


def test_exchange_does_not_inherit_original_capabilities(exchanged_claims: dict) -> None:
    original = {"web_search", "content_fetch"}
    assert not original & set(exchanged_claims["aap_capabilities"]), (
        "Exchanged token must not carry the originating agent's capabilities."
    )


def test_data_sensitivity_threads_through_exchange(
    claims: dict, exchanged_claims: dict
) -> None:
    assert (
        exchanged_claims["aap_task"]["data_sensitivity"]
        == claims["aap_task"]["data_sensitivity"]
        == "internal"
    )


def test_data_sensitivity_restricted_round_trip() -> None:
    # Spot-check a second sensitivity level so the test isn't "internal"-specific.
    t = mint_token(**{**_MINT_KWARGS, "data_sensitivity": "restricted"})
    c = validate_token(t, secret_key=_SECRET)
    assert c["aap_task"]["data_sensitivity"] == "restricted"
    ec = validate_token(exchange_token(token=t, **_EXCHANGE_KWARGS), secret_key=_SECRET)
    assert ec["aap_task"]["data_sensitivity"] == "restricted"


def test_empty_capabilities_list_is_valid() -> None:
    # An agent with no capabilities (observer role) must still produce a valid token.
    t = mint_token(**{**_MINT_KWARGS, "capabilities": []})
    c = validate_token(t, secret_key=_SECRET)
    assert c["aap_capabilities"] == []


def test_max_depth_constant_is_positive_integer() -> None:
    # MAX_DEPTH must be exported and must be a sane positive integer.
    assert isinstance(MAX_DEPTH, int)
    assert MAX_DEPTH >= 1


# ── 5. DPoP meta-test ─────────────────────────────────────────────────────────

_TOKEN_SERVICE_PATH = Path(__file__).parent.parent / "src" / "auth" / "token_service.py"


def test_token_service_file_exists() -> None:
    assert _TOKEN_SERVICE_PATH.exists(), (
        "src/auth/token_service.py does not exist — "
        "the AAP token service implementation is missing."
    )


def test_token_service_contains_dpop_todo_comments() -> None:
    """
    token_service.py must retain [DPOP-TODO] markers wherever DPoP-bound
    exchange calls, token binding, or cnf claim injection will be wired in.
    This test is a guardrail: it ensures the implementation roadmap markers
    cannot be silently deleted during refactors, keeping the DPoP migration
    path visible in source until each item is completed.

    See DPOP_IMPLEMENTATION_GUIDE.md for the full DPoP rollout plan.
    """
    assert _TOKEN_SERVICE_PATH.exists(), "src/auth/token_service.py not found"
    source = _TOKEN_SERVICE_PATH.read_text(encoding="utf-8")
    assert "[DPOP-TODO]" in source, (
        "src/auth/token_service.py must contain at least one [DPOP-TODO] comment "
        "marking where DPoP-bound A2A authentication needs to be wired in "
        "(e.g. cnf claim injection on mint, DPoP-proof validation on exchange). "
        "Remove this assertion only once every [DPOP-TODO] in that file is resolved."
    )
