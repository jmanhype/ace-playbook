"""Contract tests for ExecutionProvider implementations.

These tests verify that all ExecutionProvider implementations correctly
implement the protocol interface defined in blackice.adapters.execution.base.

Per FR-018: Provider interfaces must have contract tests ensuring
consistent behavior across all implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Type

import pytest

from blackice.adapters.execution.base import (
    BaseExecutionProvider,
    ExecutionCapabilities,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionProvider,
    ExecutionResult,
    HealthStatus,
)
from blackice.adapters.execution.local import LocalExecutionProvider
from blackice.adapters.execution.selector import (
    ProviderSelector,
    SelectionRequirements,
    SelectionStrategy,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockExecutionProvider(BaseExecutionProvider):
    """Mock provider for testing selector behavior."""

    def __init__(
        self,
        name: str = "mock",
        environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
        healthy: bool = True,
        supports_isolation: bool = False,
        supports_streaming: bool = True,
        supports_attach: bool = False,
    ) -> None:
        self._name = name
        self._healthy = healthy
        self._caps = ExecutionCapabilities(
            environment=environment,
            supports_isolation=supports_isolation,
            supports_streaming=supports_streaming,
            supports_attach=supports_attach,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return self._caps

    async def execute(
        self,
        command: str | list[str],
        *,
        config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        cmd_str = command if isinstance(command, str) else " ".join(command)
        return ExecutionResult(
            exit_code=0,
            stdout=f"mock output for: {cmd_str}",
            stderr="",
            duration_seconds=0.1,
            command=cmd_str,
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._healthy,
            environment=self._caps.environment,
            latency_ms=5.0 if self._healthy else None,
            error=None if self._healthy else "Mock unhealthy",
        )


@pytest.fixture
def local_provider() -> LocalExecutionProvider:
    """Create a local execution provider for testing."""
    return LocalExecutionProvider()


@pytest.fixture
def mock_providers() -> list[MockExecutionProvider]:
    """Create a set of mock providers with different capabilities."""
    return [
        MockExecutionProvider(
            name="local-mock",
            environment=ExecutionEnvironment.LOCAL,
            healthy=True,
            supports_isolation=False,
        ),
        MockExecutionProvider(
            name="container-mock",
            environment=ExecutionEnvironment.CONTAINER,
            healthy=True,
            supports_isolation=True,
        ),
        MockExecutionProvider(
            name="sandbox-mock",
            environment=ExecutionEnvironment.SANDBOX,
            healthy=True,
            supports_isolation=True,
        ),
    ]


# =============================================================================
# Protocol Contract Tests
# =============================================================================


class TestExecutionProviderProtocol:
    """Contract tests that all ExecutionProvider implementations must pass."""

    @pytest.mark.asyncio
    async def test_name_property_returns_string(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """All providers must have a string name."""
        assert isinstance(local_provider.name, str)
        assert len(local_provider.name) > 0

    @pytest.mark.asyncio
    async def test_capabilities_returns_valid_object(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """All providers must return valid ExecutionCapabilities."""
        caps = local_provider.capabilities
        assert isinstance(caps, ExecutionCapabilities)
        assert isinstance(caps.environment, ExecutionEnvironment)
        assert isinstance(caps.supports_streaming, bool)
        assert isinstance(caps.supports_isolation, bool)
        assert isinstance(caps.supports_attach, bool)
        assert caps.max_timeout > 0
        assert caps.max_concurrent > 0

    @pytest.mark.asyncio
    async def test_execute_returns_result(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Execute must return a valid ExecutionResult."""
        result = await local_provider.execute("echo hello")
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.duration_seconds, float)
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_execute_successful_command(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Successful commands should have exit code 0."""
        result = await local_provider.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_failing_command(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Failing commands should have non-zero exit code."""
        result = await local_provider.execute("false")  # Unix command that always fails
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_execute_with_list_command(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Providers must accept both string and list commands."""
        result = await local_provider.execute(["echo", "hello", "world"])
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert "world" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_config(
        self, local_provider: LocalExecutionProvider, tmp_path: Path
    ) -> None:
        """Providers must respect ExecutionConfig."""
        config = ExecutionConfig(
            working_dir=tmp_path,
            env={"TEST_VAR": "test_value"},
            timeout=60.0,
            shell=True,  # Required for env var expansion
        )
        result = await local_provider.execute(
            "echo $TEST_VAR",
            config=config,
        )
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_health_returns_status(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Health must return a valid HealthStatus."""
        status = await local_provider.health()
        assert isinstance(status, HealthStatus)
        assert isinstance(status.healthy, bool)
        assert isinstance(status.environment, ExecutionEnvironment)


# =============================================================================
# Provider Selector Tests
# =============================================================================


class TestProviderSelector:
    """Tests for the ProviderSelector capability negotiation."""

    @pytest.mark.asyncio
    async def test_select_returns_first_healthy(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """Selector should return first healthy provider."""
        selector = ProviderSelector(mock_providers)
        result = await selector.select()
        assert result.provider is not None
        assert result.provider.name == "local-mock"

    @pytest.mark.asyncio
    async def test_select_with_isolation_requirement(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """Selector should filter by isolation requirement."""
        selector = ProviderSelector(mock_providers)
        result = await selector.select(
            SelectionRequirements(requires_isolation=True)
        )
        assert result.provider is not None
        assert result.provider.capabilities.supports_isolation is True

    @pytest.mark.asyncio
    async def test_select_prefer_isolated_strategy(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """PREFER_ISOLATED strategy should prioritize sandbox/container."""
        selector = ProviderSelector(
            mock_providers,
            strategy=SelectionStrategy.PREFER_ISOLATED,
        )
        result = await selector.select()
        assert result.provider is not None
        assert result.provider.name == "sandbox-mock"

    @pytest.mark.asyncio
    async def test_select_skips_unhealthy(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """Selector should skip unhealthy providers."""
        mock_providers[0]._healthy = False
        selector = ProviderSelector(mock_providers)
        result = await selector.select()
        assert result.provider is not None
        assert result.provider.name != "local-mock"

    @pytest.mark.asyncio
    async def test_select_no_healthy_providers(self) -> None:
        """Selector should return None when no providers are healthy."""
        providers = [
            MockExecutionProvider(name="unhealthy", healthy=False),
        ]
        selector = ProviderSelector(providers)
        result = await selector.select()
        assert result.provider is None
        assert "unhealthy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_select_with_environment_filter(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """Selector should respect environment filters."""
        selector = ProviderSelector(mock_providers)
        result = await selector.select(
            SelectionRequirements(
                allowed_environments={ExecutionEnvironment.CONTAINER}
            )
        )
        assert result.provider is not None
        assert result.provider.capabilities.environment == ExecutionEnvironment.CONTAINER

    @pytest.mark.asyncio
    async def test_register_and_unregister(self) -> None:
        """Selector should support dynamic provider registration."""
        selector = ProviderSelector([])
        provider = MockExecutionProvider(name="dynamic")

        assert len(selector.providers) == 0
        selector.register(provider)
        assert len(selector.providers) == 1

        removed = selector.unregister(provider)
        assert removed is True
        assert len(selector.providers) == 0

    @pytest.mark.asyncio
    async def test_get_health_report(
        self, mock_providers: list[MockExecutionProvider]
    ) -> None:
        """Selector should report health of all providers."""
        selector = ProviderSelector(mock_providers)
        report = await selector.get_health_report()
        assert len(report) == 3
        assert all(isinstance(s, HealthStatus) for s in report.values())

    @pytest.mark.asyncio
    async def test_select_with_fallback_raises(self) -> None:
        """select_with_fallback should raise when no provider available."""
        selector = ProviderSelector([])
        with pytest.raises(RuntimeError):
            await selector.select_with_fallback()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestExecutionProviderEdgeCases:
    """Edge case tests for execution providers."""

    @pytest.mark.asyncio
    async def test_empty_command(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Providers should handle empty commands gracefully."""
        result = await local_provider.execute("")
        # Should either succeed (no-op) or fail gracefully
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_command_with_special_characters(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Providers should handle special characters safely."""
        result = await local_provider.execute('echo "hello; world"')
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_concurrent_executions(
        self, local_provider: LocalExecutionProvider
    ) -> None:
        """Providers should handle concurrent executions."""
        import asyncio

        commands = [f"echo {i}" for i in range(5)]
        results = await asyncio.gather(
            *[local_provider.execute(cmd) for cmd in commands]
        )
        assert len(results) == 5
        assert all(r.exit_code == 0 for r in results)
