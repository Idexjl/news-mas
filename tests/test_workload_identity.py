"""
Tests for src.auth.workload_identity — workload identity providers, config, and factory.

Covers:
  - LocalDevIdentityProvider: get_identity_token, exchange_for_aap_token,
    get_workload_claims, validate_incoming_token
  - WorkloadIdentityConfig: reads WORKLOAD_IDENTITY_PROVIDER from environment
  - get_identity_provider(): returns correct concrete type
  - Unknown provider raises clear ValueError
  - Known-but-unimplemented providers raise NotImplementedError
  - Meta-test: [WORKLOAD-IDENTITY-TODO] markers are present in source
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.auth.workload_identity import (
    LocalDevIdentityProvider,
    WorkloadIdentityConfig,
    WorkloadIdentityProvider,
    _VALID_PROVIDERS,
    get_identity_provider,
)

_WORKLOAD_IDENTITY_PATH = (
    Path(__file__).parent.parent / "src" / "auth" / "workload_identity.py"
)

SECRET = "test-secret-key"


# ── LocalDevIdentityProvider ──────────────────────────────────────────────────

class TestLocalDevIdentityProvider:
    def setup_method(self):
        self.provider = LocalDevIdentityProvider(secret_key=SECRET)

    def test_get_identity_token_returns_nonempty_string(self):
        token = self.provider.get_identity_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_get_identity_token_is_three_segment_jwt(self):
        token = self.provider.get_identity_token()
        assert token.count(".") == 2, "Expected header.payload.signature format"

    def test_exchange_for_aap_token_returns_valid_aap_token(self):
        identity_token = self.provider.get_identity_token()
        aap_token = self.provider.exchange_for_aap_token(
            identity_token, capabilities=["search", "score"]
        )
        assert isinstance(aap_token, str)
        assert aap_token.count(".") == 2

    def test_exchange_for_aap_token_contains_all_aap_sections(self):
        import base64
        import json

        identity_token = self.provider.get_identity_token()
        aap_token = self.provider.exchange_for_aap_token(
            identity_token, capabilities=["search"]
        )
        _, payload_b64, _ = aap_token.split(".")
        pad = 4 - len(payload_b64) % 4
        if pad != 4:
            payload_b64 += "=" * pad
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))

        for section in (
            "aap_agent", "aap_task", "aap_capabilities",
            "aap_oversight", "aap_delegation", "aap_context", "aap_audit",
        ):
            assert section in claims, f"AAP section '{section}' missing from exchanged token"

    def test_exchange_for_aap_token_preserves_capabilities(self):
        import base64
        import json

        caps = ["read", "write", "score"]
        identity_token = self.provider.get_identity_token()
        aap_token = self.provider.exchange_for_aap_token(identity_token, capabilities=caps)
        _, payload_b64, _ = aap_token.split(".")
        pad = 4 - len(payload_b64) % 4
        if pad != 4:
            payload_b64 += "=" * pad
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        assert claims["aap_capabilities"] == caps

    def test_get_workload_claims_returns_expected_keys(self):
        claims = self.provider.get_workload_claims()
        assert set(claims.keys()) == {"agent_id", "environment", "region", "provider_type"}

    def test_get_workload_claims_provider_type_is_local(self):
        claims = self.provider.get_workload_claims()
        assert claims["provider_type"] == "local"

    def test_get_workload_claims_region_is_local(self):
        claims = self.provider.get_workload_claims()
        assert claims["region"] == "local"

    def test_validate_incoming_token_accepts_correct_secret(self):
        # Should not raise.
        self.provider.validate_incoming_token(SECRET)

    def test_validate_incoming_token_rejects_wrong_secret(self):
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            self.provider.validate_incoming_token("wrong-secret")

    def test_validate_incoming_token_no_op_when_secret_empty(self):
        provider = LocalDevIdentityProvider(secret_key="")
        provider.validate_incoming_token("anything")  # must not raise

    def test_exchange_rejects_tampered_identity_token(self):
        identity_token = self.provider.get_identity_token()
        parts = identity_token.split(".")
        tampered = f"{parts[0]}.{parts[1]}.badsignature"
        with pytest.raises(ValueError):
            self.provider.exchange_for_aap_token(tampered, capabilities=[])

    def test_agent_id_defaults_to_local_dev_agent(self):
        provider = LocalDevIdentityProvider(secret_key=SECRET)
        claims = provider.get_workload_claims()
        assert claims["agent_id"] == "local-dev-agent"

    def test_agent_id_can_be_overridden(self):
        provider = LocalDevIdentityProvider(secret_key=SECRET, agent_id="heat-scorer")
        claims = provider.get_workload_claims()
        assert claims["agent_id"] == "heat-scorer"

    def test_reads_secret_from_env_when_not_passed(self):
        with patch.dict(os.environ, {"MAS_SECRET_KEY": SECRET}):
            provider = LocalDevIdentityProvider()
        # Should be able to validate with the key it loaded from env.
        provider.validate_incoming_token(SECRET)

    def test_is_workload_identity_provider_subclass(self):
        assert isinstance(self.provider, WorkloadIdentityProvider)


# ── WorkloadIdentityConfig ────────────────────────────────────────────────────

class TestWorkloadIdentityConfig:
    def test_defaults_to_local_when_env_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "WORKLOAD_IDENTITY_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            config = WorkloadIdentityConfig()
        assert config.provider_type == "local"

    def test_reads_provider_type_from_environment(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "local"}):
            config = WorkloadIdentityConfig()
        assert config.provider_type == "local"

    def test_strips_whitespace_from_env_value(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "  local  "}):
            config = WorkloadIdentityConfig()
        assert config.provider_type == "local"

    def test_get_provider_returns_local_dev_provider(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "local"}):
            provider = WorkloadIdentityConfig().get_provider()
        assert isinstance(provider, LocalDevIdentityProvider)


# ── get_identity_provider factory ────────────────────────────────────────────

class TestGetIdentityProvider:
    def test_returns_local_dev_provider_for_local(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "local"}):
            provider = get_identity_provider()
        assert isinstance(provider, LocalDevIdentityProvider)

    def test_unknown_provider_raises_value_error(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "nonexistent-cloud"}):
            with pytest.raises(ValueError, match="nonexistent-cloud"):
                get_identity_provider()

    def test_unknown_provider_error_lists_valid_options(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "foo"}):
            with pytest.raises(ValueError) as exc_info:
                get_identity_provider()
        msg = str(exc_info.value)
        for valid in _VALID_PROVIDERS:
            assert valid in msg, f"Error message missing valid option '{valid}'"

    def test_azure_managed_identity_raises_not_implemented(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "azure-managed-identity"}):
            with pytest.raises(NotImplementedError, match="[Aa]zure"):
                get_identity_provider()

    def test_aws_sts_raises_not_implemented(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "aws-sts"}):
            with pytest.raises(NotImplementedError, match="[Aa][Ww][Ss]"):
                get_identity_provider()

    def test_gcp_workload_raises_not_implemented(self):
        with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": "gcp-workload"}):
            with pytest.raises(NotImplementedError, match="[Gg][Cc][Pp]"):
                get_identity_provider()

    def test_not_implemented_error_points_to_source_file(self):
        """NotImplementedError messages must reference the source file so devs know where to look."""
        for provider in ("azure-managed-identity", "aws-sts", "gcp-workload"):
            with patch.dict(os.environ, {"WORKLOAD_IDENTITY_PROVIDER": provider}):
                with pytest.raises(NotImplementedError) as exc_info:
                    get_identity_provider()
            assert "workload_identity.py" in str(exc_info.value), (
                f"NotImplementedError for '{provider}' must reference workload_identity.py"
            )


# ── Meta-test: [WORKLOAD-IDENTITY-TODO] markers ───────────────────────────────

def test_workload_identity_contains_todo_comments() -> None:
    """
    src/auth/workload_identity.py must retain [WORKLOAD-IDENTITY-TODO] markers
    for every production provider that is not yet implemented (azure-managed-identity,
    aws-sts, gcp-workload) and for the LocalDevIdentityProvider replacement note.

    Remove or update this assertion only once the corresponding provider is
    fully implemented and its TODO comment is resolved.
    """
    source = _WORKLOAD_IDENTITY_PATH.read_text(encoding="utf-8")
    assert "[WORKLOAD-IDENTITY-TODO]" in source, (
        "src/auth/workload_identity.py must contain at least one "
        "[WORKLOAD-IDENTITY-TODO] comment. These mark unimplemented production "
        "providers and the LocalDevIdentityProvider replacement point. "
        "Remove this assertion only once every TODO in that file is resolved."
    )


def test_workload_identity_todo_comments_cover_all_providers() -> None:
    """Each unimplemented provider must have its own [WORKLOAD-IDENTITY-TODO] block."""
    source = _WORKLOAD_IDENTITY_PATH.read_text(encoding="utf-8")
    expected_markers = [
        "AzureManagedIdentityProvider",
        "AWSSTSFederationProvider",
        "GCPWorkloadIdentityProvider",
        "GitHubActions",
    ]
    for marker in expected_markers:
        assert marker in source, (
            f"Expected '{marker}' to appear in a [WORKLOAD-IDENTITY-TODO] block "
            "in src/auth/workload_identity.py. "
            "Add the implementation guide comment before removing this assertion."
        )
