"""Provider Selector for BLACKICE 3.0.

Implements capability negotiation and provider selection for execution
environments based on requirements, availability, and preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from blackice.adapters.execution.base import (
    ExecutionCapabilities,
    ExecutionEnvironment,
    ExecutionProvider,
    HealthStatus,
)
from blackice.instrumentation import get_logger

logger = get_logger(__name__)


class SelectionStrategy(str, Enum):
    """Strategy for selecting providers."""

    PREFER_LOCAL = "prefer_local"  # Local first, then containers, then remote
    PREFER_ISOLATED = "prefer_isolated"  # Sandboxed/container first
    PREFER_FASTEST = "prefer_fastest"  # Based on latency checks
    FAILOVER = "failover"  # First available from ordered list
    ROUND_ROBIN = "round_robin"  # Distribute across providers


@dataclass
class SelectionRequirements:
    """Requirements for provider selection."""

    # Minimum capabilities
    requires_isolation: bool = False
    requires_streaming: bool = False
    requires_attach: bool = False

    # Resource requirements
    min_timeout: float = 60.0
    min_concurrent: int = 1

    # Environment preferences
    allowed_environments: set[ExecutionEnvironment] = field(
        default_factory=lambda: {
            ExecutionEnvironment.LOCAL,
            ExecutionEnvironment.CONTAINER,
            ExecutionEnvironment.SANDBOX,
        }
    )
    excluded_environments: set[ExecutionEnvironment] = field(default_factory=set)


@dataclass
class SelectionResult:
    """Result of provider selection."""

    provider: ExecutionProvider | None
    reason: str
    fallback_providers: list[ExecutionProvider] = field(default_factory=list)
    checked_providers: list[str] = field(default_factory=list)


class ProviderSelector:
    """Selects execution providers based on requirements and availability.

    Implements capability negotiation (FR-018) to dynamically select
    the most appropriate execution environment.

    Example:
        ```python
        selector = ProviderSelector(
            providers=[local_provider, container_provider, sandbox_provider],
            strategy=SelectionStrategy.PREFER_ISOLATED,
        )

        # Select based on requirements
        result = await selector.select(
            SelectionRequirements(requires_isolation=True)
        )
        if result.provider:
            await result.provider.execute("pytest tests/")
        ```
    """

    def __init__(
        self,
        providers: Sequence[ExecutionProvider],
        strategy: SelectionStrategy = SelectionStrategy.FAILOVER,
        health_check_timeout: float = 5.0,
    ) -> None:
        """Initialize the selector.

        Args:
            providers: Available execution providers
            strategy: Selection strategy to use
            health_check_timeout: Timeout for health checks
        """
        self._providers = list(providers)
        self._strategy = strategy
        self._health_check_timeout = health_check_timeout
        self._round_robin_index = 0

    @property
    def providers(self) -> list[ExecutionProvider]:
        """Get registered providers."""
        return self._providers.copy()

    def register(self, provider: ExecutionProvider) -> None:
        """Register a new provider.

        Args:
            provider: Provider to register
        """
        if provider not in self._providers:
            self._providers.append(provider)
            logger.info(
                "provider_registered",
                name=provider.name,
                environment=provider.capabilities.environment.value,
            )

    def unregister(self, provider: ExecutionProvider) -> bool:
        """Unregister a provider.

        Args:
            provider: Provider to unregister

        Returns:
            True if provider was found and removed
        """
        try:
            self._providers.remove(provider)
            logger.info("provider_unregistered", name=provider.name)
            return True
        except ValueError:
            return False

    async def select(
        self,
        requirements: SelectionRequirements | None = None,
    ) -> SelectionResult:
        """Select the best provider based on requirements.

        Args:
            requirements: Selection requirements (uses defaults if None)

        Returns:
            SelectionResult with chosen provider and metadata
        """
        requirements = requirements or SelectionRequirements()

        # Filter by capabilities first
        candidates = self._filter_by_capabilities(requirements)

        if not candidates:
            return SelectionResult(
                provider=None,
                reason="No providers match required capabilities",
                checked_providers=[p.name for p in self._providers],
            )

        # Order by strategy
        ordered = self._order_by_strategy(candidates)

        # Check health and return first healthy
        return await self._select_first_healthy(ordered)

    def _filter_by_capabilities(
        self,
        requirements: SelectionRequirements,
    ) -> list[ExecutionProvider]:
        """Filter providers by capability requirements."""
        candidates = []

        for provider in self._providers:
            caps = provider.capabilities

            # Check environment restrictions
            if caps.environment in requirements.excluded_environments:
                continue
            if caps.environment not in requirements.allowed_environments:
                continue

            # Check capability requirements
            if requirements.requires_isolation and not caps.supports_isolation:
                continue
            if requirements.requires_streaming and not caps.supports_streaming:
                continue
            if requirements.requires_attach and not caps.supports_attach:
                continue

            # Check resource requirements
            if caps.max_timeout < requirements.min_timeout:
                continue
            if caps.max_concurrent < requirements.min_concurrent:
                continue

            candidates.append(provider)

        return candidates

    def _order_by_strategy(
        self,
        providers: list[ExecutionProvider],
    ) -> list[ExecutionProvider]:
        """Order providers by selection strategy."""
        if self._strategy == SelectionStrategy.PREFER_LOCAL:
            # Local > Container > Sandbox > Remote
            priority = {
                ExecutionEnvironment.LOCAL: 0,
                ExecutionEnvironment.CONTAINER: 1,
                ExecutionEnvironment.SANDBOX: 2,
                ExecutionEnvironment.REMOTE: 3,
            }
            return sorted(
                providers,
                key=lambda p: priority.get(p.capabilities.environment, 99),
            )

        elif self._strategy == SelectionStrategy.PREFER_ISOLATED:
            # Sandbox > Container > Local > Remote
            priority = {
                ExecutionEnvironment.SANDBOX: 0,
                ExecutionEnvironment.CONTAINER: 1,
                ExecutionEnvironment.LOCAL: 2,
                ExecutionEnvironment.REMOTE: 3,
            }
            return sorted(
                providers,
                key=lambda p: priority.get(p.capabilities.environment, 99),
            )

        elif self._strategy == SelectionStrategy.ROUND_ROBIN:
            # Rotate through providers
            n = len(providers)
            if n == 0:
                return providers
            idx = self._round_robin_index % n
            self._round_robin_index += 1
            return providers[idx:] + providers[:idx]

        # FAILOVER or PREFER_FASTEST: keep original order
        return providers

    async def _select_first_healthy(
        self,
        providers: list[ExecutionProvider],
    ) -> SelectionResult:
        """Select first healthy provider from ordered list."""
        checked: list[str] = []
        fallbacks: list[ExecutionProvider] = []

        for provider in providers:
            checked.append(provider.name)

            try:
                health = await provider.health()
                if health.healthy:
                    # Found healthy provider
                    return SelectionResult(
                        provider=provider,
                        reason=f"Selected {provider.name} ({health.environment.value})",
                        fallback_providers=fallbacks,
                        checked_providers=checked,
                    )
                else:
                    logger.warning(
                        "provider_unhealthy",
                        name=provider.name,
                        error=health.error,
                    )
                    fallbacks.append(provider)
            except Exception as e:
                logger.warning(
                    "provider_health_check_failed",
                    name=provider.name,
                    error=str(e),
                )
                fallbacks.append(provider)

        return SelectionResult(
            provider=None,
            reason="All matching providers are unhealthy",
            fallback_providers=fallbacks,
            checked_providers=checked,
        )

    async def select_with_fallback(
        self,
        requirements: SelectionRequirements | None = None,
    ) -> ExecutionProvider:
        """Select provider, raising if none available.

        Args:
            requirements: Selection requirements

        Returns:
            Selected provider

        Raises:
            RuntimeError: If no healthy provider is available
        """
        result = await self.select(requirements)
        if result.provider is None:
            raise RuntimeError(f"No execution provider available: {result.reason}")
        return result.provider

    async def get_all_healthy(self) -> list[ExecutionProvider]:
        """Get all currently healthy providers.

        Returns:
            List of healthy providers
        """
        healthy = []
        for provider in self._providers:
            try:
                health = await provider.health()
                if health.healthy:
                    healthy.append(provider)
            except Exception:
                pass
        return healthy

    async def get_health_report(self) -> dict[str, HealthStatus]:
        """Get health status of all providers.

        Returns:
            Dict mapping provider names to health status
        """
        report = {}
        for provider in self._providers:
            try:
                health = await provider.health()
                report[provider.name] = health
            except Exception as e:
                report[provider.name] = HealthStatus(
                    healthy=False,
                    environment=provider.capabilities.environment,
                    error=str(e),
                )
        return report


__all__ = [
    "SelectionStrategy",
    "SelectionRequirements",
    "SelectionResult",
    "ProviderSelector",
]
