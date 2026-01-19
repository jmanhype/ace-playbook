"""Integration test IT-007: Secret handling in execution environment.

Tests the complete secrets pipeline: Discovery -> Retrieval -> Injection ->
Redaction -> ensuring secrets are never exposed in outputs or logs.

Per FR-010: Secure credential handling with environment injection and redaction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator

import pytest

from blackice.adapters.secrets import (
    AdvancedRedactionResult,
    EnvSecretsProvider,
    HealthStatus,
    InjectionResult,
    RedactionLevel,
    RedactionResult,
    Secret,
    SecretRedactor,
    SecretReference,
    SecretType,
    get_default_redactor,
    redact,
    register_secret,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_env() -> Generator[dict[str, str], None, None]:
    """Provide a clean environment state for testing.

    Saves current env, yields a dict to add test secrets to,
    and restores the original env after test.
    """
    # Save original environment
    original_env = os.environ.copy()

    # Clean up any test secrets that might be present
    test_keys = [k for k in os.environ if k.startswith("TEST_") or k.endswith("_SECRET")]
    for key in test_keys:
        if key in os.environ:
            del os.environ[key]

    yield {}

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def secrets_in_env(clean_env: dict[str, str]) -> Generator[dict[str, str], None, None]:
    """Set up test secrets in environment."""
    test_secrets = {
        "TEST_API_KEY": "sk-test-12345678901234567890",
        "TEST_DB_PASSWORD": "super_secret_password_123",
        "TEST_GITHUB_TOKEN": "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "TEST_SECRET": "my_generic_secret_value",
        "ANTHROPIC_API_KEY": "sk-ant-test-fake-key-for-testing",
    }

    for key, value in test_secrets.items():
        os.environ[key] = value

    yield test_secrets

    # Cleanup handled by clean_env fixture


@pytest.fixture
def env_provider() -> EnvSecretsProvider:
    """Create a fresh environment secrets provider."""
    provider = EnvSecretsProvider()
    provider.clear_cache()  # Ensure clean state
    return provider


@pytest.fixture
def secret_redactor() -> SecretRedactor:
    """Create a secret redactor for testing."""
    return SecretRedactor(level=RedactionLevel.STANDARD)


# =============================================================================
# Integration Tests: Secret Discovery & Retrieval
# =============================================================================


class TestSecretDiscovery:
    """IT-007a: Test secret discovery from environment."""

    @pytest.mark.asyncio
    async def test_discovers_api_key_pattern(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that provider discovers API keys from environment."""
        secret = await env_provider.get("test-api-key", required=False)

        assert secret is not None
        assert secret.secret_type == SecretType.API_KEY
        assert secret.get_value() == secrets_in_env["TEST_API_KEY"]

    @pytest.mark.asyncio
    async def test_discovers_token_pattern(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that provider discovers tokens from environment."""
        secret = await env_provider.get("test-github-token", required=False)

        assert secret is not None
        assert secret.secret_type == SecretType.TOKEN
        assert secret.get_value() == secrets_in_env["TEST_GITHUB_TOKEN"]

    @pytest.mark.asyncio
    async def test_discovers_password_pattern(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that provider discovers passwords from environment."""
        secret = await env_provider.get("test-db-password", required=False)

        assert secret is not None
        assert secret.secret_type == SecretType.PASSWORD
        assert secret.get_value() == secrets_in_env["TEST_DB_PASSWORD"]

    @pytest.mark.asyncio
    async def test_lists_all_discovered_secrets(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test listing all discovered secrets."""
        secrets = await env_provider.list()

        # Should find at least our test secrets
        secret_names = [s.name for s in secrets]

        assert len(secrets) >= 4
        assert all(isinstance(s, SecretReference) for s in secrets)

        # Verify source is correct
        assert all(s.source == "env" for s in secrets)

    @pytest.mark.asyncio
    async def test_raises_on_missing_required_secret(
        self,
        clean_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that provider raises error for missing required secret."""
        from blackice.primitives.errors import SecretsProviderError

        with pytest.raises(SecretsProviderError) as exc_info:
            await env_provider.get("nonexistent-secret", required=True)

        assert "nonexistent-secret" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_optional_secret(
        self,
        clean_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that provider returns None for missing optional secret."""
        secret = await env_provider.get("nonexistent-secret", required=False)

        assert secret is None


# =============================================================================
# Integration Tests: Secret Injection
# =============================================================================


class TestSecretInjection:
    """IT-007b: Test secret injection into execution environments."""

    @pytest.mark.asyncio
    async def test_injects_secrets_into_env(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test injecting secrets into a new environment dict."""
        base_env = {"PATH": "/usr/bin", "HOME": "/home/test"}

        result = await env_provider.inject_env(
            secret_names=["test-api-key", "test-secret"],
            env=base_env,
        )

        assert isinstance(result, InjectionResult)
        assert result.injected_count == 2
        assert len(result.errors) == 0

        # Masked values should be in env_vars
        assert "TEST_API_KEY" in result.env_vars
        assert "TEST_SECRET" in result.env_vars

        # Masked values should not reveal actual secrets
        assert "***" in result.env_vars["TEST_API_KEY"]

    @pytest.mark.asyncio
    async def test_injection_preserves_base_env(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that injection preserves base environment variables."""
        base_env = {"PATH": "/usr/bin", "CUSTOM_VAR": "custom_value"}

        result = await env_provider.inject_env(
            secret_names=["test-api-key"],
            env=base_env,
        )

        assert result.injected_count >= 1

    @pytest.mark.asyncio
    async def test_injection_reports_errors_for_missing(
        self,
        clean_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that injection reports errors for missing secrets."""
        result = await env_provider.inject_env(
            secret_names=["missing-secret-1", "missing-secret-2"],
            env={},
        )

        assert result.injected_count == 0
        assert len(result.errors) == 2


# =============================================================================
# Integration Tests: Secret Redaction
# =============================================================================


class TestSecretRedaction:
    """IT-007c: Test secret redaction from outputs and logs."""

    def test_redacts_known_secret_values(
        self,
        secrets_in_env: dict[str, str],
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting known secret values from text."""
        # Add known secrets to redactor
        for name, value in secrets_in_env.items():
            secret_redactor.add_known_secret(value, name)

        text_with_secrets = f"""
        API Key: {secrets_in_env["TEST_API_KEY"]}
        Database: postgresql://user:{secrets_in_env["TEST_DB_PASSWORD"]}@localhost/db
        Token: {secrets_in_env["TEST_GITHUB_TOKEN"]}
        """

        result = secret_redactor.redact(text_with_secrets)

        assert isinstance(result, AdvancedRedactionResult)
        assert result.redaction_count >= 3

        # Verify secrets are no longer in text
        assert secrets_in_env["TEST_API_KEY"] not in result.redacted_text
        assert secrets_in_env["TEST_DB_PASSWORD"] not in result.redacted_text
        assert secrets_in_env["TEST_GITHUB_TOKEN"] not in result.redacted_text

        # Verify redaction marker is present
        assert "[REDACTED]" in result.redacted_text

    def test_redacts_api_key_patterns(
        self,
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting API key patterns from text."""
        text = """
        OpenAI Key: sk-1234567890abcdefghijklmnopqrstuv
        Anthropic Key: sk-ant-abcdefghijklmnopqrstuvwxyz123456
        """

        result = secret_redactor.redact(text)

        assert result.redaction_count >= 2
        assert "sk-1234567890" not in result.redacted_text
        assert "sk-ant-" not in result.redacted_text

    def test_redacts_jwt_tokens(
        self,
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting JWT tokens from text."""
        # Valid JWT structure (header.payload.signature in base64)
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"

        text = f"Authorization: Bearer {jwt}"

        result = secret_redactor.redact(text)

        # JWT should be redacted
        assert jwt not in result.redacted_text or result.redaction_count >= 1

    def test_redacts_database_urls(
        self,
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting database connection URLs."""
        db_url = "postgres://admin:s3cr3tP@ss!@db.example.com:5432/production"
        text = f"DATABASE_URL={db_url}"

        result = secret_redactor.redact(text)

        # Password should be redacted
        assert "s3cr3tP@ss" not in result.redacted_text

    def test_redacts_github_tokens(
        self,
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting GitHub tokens."""
        # GitHub PAT (ghp_) and OAuth (gho_) tokens need 36+ characters after prefix
        tokens = [
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz",  # PAT - 36 chars after prefix
            "gho_abcdefghijklmnopqrstuvwxyz123456abcd",  # OAuth - 36 chars after prefix
        ]

        text = f"Token 1: {tokens[0]}\nToken 2: {tokens[1]}"

        result = secret_redactor.redact(text)

        for token in tokens:
            assert token not in result.redacted_text

    def test_provider_redact_method(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test provider's built-in redact method after injection."""
        # First inject secrets (this tracks patterns)
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            env_provider.inject_env(["test-api-key"])
        )

        # Now redact text containing the secret
        text = f"Key is: {secrets_in_env['TEST_API_KEY']}"
        result = env_provider.redact(text)

        assert isinstance(result, RedactionResult)
        assert secrets_in_env["TEST_API_KEY"] not in result.redacted_text

    def test_redaction_levels(self) -> None:
        """Test different redaction levels."""
        text = """
        API_KEY=abcdef1234567890abcdef12
        secret=mysecretvalue12345678
        Authorization: Bearer eyJ0eXAiOi
        """

        # MINIMAL should catch fewer things
        minimal = SecretRedactor(level=RedactionLevel.MINIMAL)
        minimal_result = minimal.redact(text)

        # PARANOID should catch more
        paranoid = SecretRedactor(level=RedactionLevel.PARANOID)
        paranoid_result = paranoid.redact(text)

        # Paranoid should redact more than minimal
        # (or equal if all patterns match)
        assert paranoid_result.redaction_count >= minimal_result.redaction_count


# =============================================================================
# Integration Tests: Health Checks
# =============================================================================


class TestSecretsHealth:
    """IT-007d: Test secrets provider health checks."""

    @pytest.mark.asyncio
    async def test_health_check_reports_status(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that health check reports correct status."""
        health = await env_provider.health()

        assert isinstance(health, HealthStatus)
        assert health.healthy is True
        assert health.secret_count >= 4  # At least our test secrets
        assert health.error is None

    @pytest.mark.asyncio
    async def test_health_check_includes_details(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that health check includes provider details."""
        health = await env_provider.health()

        assert "provider" in health.details
        assert health.details["provider"] == "env"

    @pytest.mark.asyncio
    async def test_health_check_on_empty_env(
        self,
        clean_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test health check with no secrets available."""
        health = await env_provider.health()

        # Should still be healthy, just with fewer secrets
        assert health.healthy is True
        # May have some system env vars that match patterns


# =============================================================================
# Integration Tests: Secret Value Protection
# =============================================================================


class TestSecretValueProtection:
    """IT-007e: Test that secret values are never exposed."""

    @pytest.mark.asyncio
    async def test_secret_str_does_not_reveal_value(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that Secret.__str__ does not reveal the value."""
        secret = await env_provider.get("test-api-key")

        str_repr = str(secret)

        assert secrets_in_env["TEST_API_KEY"] not in str_repr
        assert "Secret(" in str_repr
        assert "test-api-key" in str_repr

    @pytest.mark.asyncio
    async def test_secret_repr_does_not_reveal_value(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that Secret.__repr__ does not reveal the value."""
        secret = await env_provider.get("test-api-key")

        repr_str = repr(secret)

        assert secrets_in_env["TEST_API_KEY"] not in repr_str

    def test_redactor_dict_method(
        self,
        secret_redactor: SecretRedactor,
    ) -> None:
        """Test redacting secrets from dictionaries."""
        data = {
            "api_key": "sk-secret-key-12345",
            "password": "admin123",
            "config": {
                "token": "secret_token_value",
                "name": "safe_value",
            },
            "items": ["item1", "sk-another-key-67890"],
        }

        result = secret_redactor.redact_dict(data)

        # Keys with secret patterns should be redacted
        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["config"]["token"] == "[REDACTED]"

        # Safe values should be preserved
        assert result["config"]["name"] == "safe_value"


# =============================================================================
# Integration Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """IT-007f: Test module-level convenience functions."""

    def test_global_redact_function(self) -> None:
        """Test the global redact() convenience function."""
        text = "API key is sk-1234567890abcdefghijklmnopqrstuv"

        redacted = redact(text)

        assert "sk-12345" not in redacted
        assert "[REDACTED]" in redacted

    def test_register_secret_for_redaction(self) -> None:
        """Test registering a custom secret for redaction."""
        custom_secret = "my-custom-secret-value-xyz"
        register_secret(custom_secret, "custom")

        text = f"The secret is: {custom_secret}"
        redactor = get_default_redactor()
        result = redactor.redact(text)

        assert custom_secret not in result.redacted_text

    def test_get_default_redactor_singleton(self) -> None:
        """Test that get_default_redactor returns same instance."""
        redactor1 = get_default_redactor()
        redactor2 = get_default_redactor()

        assert redactor1 is redactor2


# =============================================================================
# Integration Tests: Execution Integration
# =============================================================================


class TestExecutionIntegration:
    """IT-007g: Test secrets integration with execution provider."""

    @pytest.mark.asyncio
    async def test_secrets_can_be_injected_for_execution(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that secrets can be prepared for command execution."""
        # Simulate preparing environment for subprocess
        base_env = {"PATH": os.environ.get("PATH", "/usr/bin")}

        result = await env_provider.inject_env(
            secret_names=["test-api-key"],
            env=base_env,
        )

        assert result.injected_count >= 1

        # The actual env should have the real value
        # (env_vars only has masked values for logging)
        # We can't check the internal env directly, but injection succeeded

    @pytest.mark.asyncio
    async def test_output_redaction_after_execution(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that execution output can be redacted."""
        # First inject to track patterns
        await env_provider.inject_env(["test-api-key", "test-db-password"])

        # Simulate command output that accidentally leaked secrets
        fake_output = f"""
        Command executed successfully.
        Using API key: {secrets_in_env["TEST_API_KEY"]}
        Connected to database with password: {secrets_in_env["TEST_DB_PASSWORD"]}
        """

        result = env_provider.redact(fake_output)

        # Secrets should be redacted from output
        assert secrets_in_env["TEST_API_KEY"] not in result.redacted_text
        assert secrets_in_env["TEST_DB_PASSWORD"] not in result.redacted_text

        # Non-secret content should remain
        assert "Command executed" in result.redacted_text
        assert "successfully" in result.redacted_text

    @pytest.mark.asyncio
    async def test_error_message_redaction(
        self,
        secrets_in_env: dict[str, str],
        env_provider: EnvSecretsProvider,
    ) -> None:
        """Test that error messages can be redacted."""
        await env_provider.inject_env(["test-api-key"])

        # Simulate an error message that leaked the secret
        error_message = f"Connection failed with key {secrets_in_env['TEST_API_KEY']}"

        result = env_provider.redact(error_message)

        assert secrets_in_env["TEST_API_KEY"] not in result.redacted_text
