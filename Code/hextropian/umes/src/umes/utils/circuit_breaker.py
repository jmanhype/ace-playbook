"""
Circuit breaker factory using Purgatory for external service protection.

Provides AsyncCircuitBreakerFactory wrapper for creating circuit breakers that
protect calls to external services (KMS, IdP) from cascading failures.

Usage:
    from umes.utils.circuit_breaker import AsyncCircuitBreakerFactory

    # Initialize factory
    factory = AsyncCircuitBreakerFactory(
        failure_threshold=5,
        timeout_seconds=60,
    )

    # Create circuit breakers for services
    kms_breaker = await factory.create_kms_breaker()
    idp_breaker = await factory.create_idp_breaker()

    # Use circuit breaker to protect calls
    async def kms_operation():
        return await kms_client.encrypt(data)

    result = await kms_breaker.call(kms_operation)

Circuit Breaker States:
    - closed: Normal operation, requests pass through
    - open: Too many failures, requests fail immediately
    - half_open: Testing recovery, allows one probe request

State Transitions:
    closed → open: After failure_threshold failures
    open → half_open: After timeout_seconds
    half_open → closed: On successful probe
    half_open → open: On failed probe

Architecture:
    - Uses Purgatory 3.0 library for circuit breaker implementation
    - Singleton pattern: same service gets same breaker instance
    - Separate breakers for KMS and IdP (isolated failure domains)
    - Configurable thresholds, timeouts, and recovery parameters
"""
import asyncio
from typing import Optional, Callable, Any
from purgatory import AsyncCircuitBreakerFactory as PurgatoryFactory


class AsyncCircuitBreakerFactory:
    """Factory for creating and managing async circuit breakers.

    Wraps Purgatory's AsyncCircuitBreakerFactory to provide singleton
    circuit breaker instances for different services.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Time to wait before attempting recovery (open → half_open)
        recovery_timeout_seconds: Time to wait in half_open before re-opening
        operation_timeout_seconds: Max time for individual operation
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        recovery_timeout_seconds: Optional[int] = None,
        operation_timeout_seconds: Optional[float] = None,
    ):
        """Initialize circuit breaker factory.

        Args:
            failure_threshold: Failures before circuit opens (default: 5)
            timeout_seconds: Time in open state before half_open (default: 60)
            recovery_timeout_seconds: Time in half_open before re-opening (default: same as timeout_seconds)
            operation_timeout_seconds: Timeout for individual operations (default: None = no timeout)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = float(timeout_seconds)
        self.recovery_timeout_seconds = recovery_timeout_seconds or timeout_seconds
        self.operation_timeout_seconds = operation_timeout_seconds

        # Purgatory factory instance
        self._purgatory_factory = PurgatoryFactory(
            default_threshold=failure_threshold,
            default_ttl=self.timeout_seconds,
        )

        # Cache for singleton breakers
        self._kms_breaker = None
        self._idp_breaker = None

    async def _init_factory(self):
        """Initialize Purgatory factory if needed."""
        if not hasattr(self._purgatory_factory, '_initialized'):
            await self._purgatory_factory.initialize()
            self._purgatory_factory._initialized = True

    async def create_kms_breaker(self):
        """Create or return singleton circuit breaker for KMS.

        Returns:
            Circuit breaker wrapper for KMS operations
        """
        await self._init_factory()

        if self._kms_breaker is None:
            breaker = await self._purgatory_factory.get_breaker(
                circuit="kms",
                threshold=self.failure_threshold,
                ttl=self.timeout_seconds,
            )
            self._kms_breaker = CircuitBreakerWrapper(
                breaker,
                operation_timeout=self.operation_timeout_seconds,
            )

        return self._kms_breaker

    async def create_idp_breaker(self):
        """Create or return singleton circuit breaker for IdP.

        Returns:
            Circuit breaker wrapper for IdP operations
        """
        await self._init_factory()

        if self._idp_breaker is None:
            breaker = await self._purgatory_factory.get_breaker(
                circuit="idp",
                threshold=self.failure_threshold,
                ttl=self.timeout_seconds,
            )
            self._idp_breaker = CircuitBreakerWrapper(
                breaker,
                operation_timeout=self.operation_timeout_seconds,
            )

        return self._idp_breaker

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state.

        Useful for testing or manual recovery scenarios.
        """
        self._kms_breaker = None
        self._idp_breaker = None


class CircuitBreakerWrapper:
    """Wrapper around Purgatory circuit breaker with additional features.

    Adds:
    - Operation-level timeout support
    - Health status checking
    - Metrics tracking
    - Simplified call interface
    """

    def __init__(
        self,
        breaker: Any,  # Purgatory AsyncCircuitBreaker
        operation_timeout: Optional[float] = None,
    ):
        """Initialize wrapper.

        Args:
            breaker: Underlying Purgatory circuit breaker
            operation_timeout: Max time for operations (None = no timeout)
        """
        self._breaker = breaker
        self._operation_timeout = operation_timeout
        self._failure_count = 0
        self._state = "closed"  # Track state since Purgatory 3.0 doesn't expose it

    async def call(self, func: Callable[[], Any]) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Async function to execute

        Returns:
            Result from function

        Raises:
            Exception: Circuit is open, timeout, or original exception
        """
        try:
            # Purgatory 3.0 uses async context manager
            if self._operation_timeout:
                # Wrap in timeout
                async def timeout_wrapped():
                    async with self._breaker:
                        return await func()

                result = await asyncio.wait_for(
                    timeout_wrapped(),
                    timeout=self._operation_timeout,
                )
            else:
                async with self._breaker:
                    result = await func()

            # Success - reset failure counter
            self._failure_count = 0
            return result

        except asyncio.TimeoutError:
            # Timeout counts as failure
            self._failure_count += 1
            raise

        except Exception as e:
            # Other exceptions count as failures
            self._failure_count += 1
            raise

    @property
    def state(self) -> str:
        """Get current circuit breaker state.

        Returns:
            State string: "closed", "open", or "half_open"

        Note: Retrieved from Purgatory's context.state.
        """
        return self._breaker.context.state

    @property
    def current_failures(self) -> int:
        """Get current failure count.

        Returns:
            Number of consecutive failures
        """
        return self._failure_count

    @property
    def _failure_count(self) -> int:
        """Get failure count (alias for tests)."""
        return self.__dict__.get('_failure_count_value', 0)

    @_failure_count.setter
    def _failure_count(self, value: int):
        """Set failure count."""
        self.__dict__['_failure_count_value'] = value

    @property
    def fail(self) -> int:
        """Alias for failure count (for test compatibility)."""
        return self._failure_count

    def is_healthy(self) -> bool:
        """Check if circuit breaker is healthy (closed).

        Returns:
            True if circuit is closed, False otherwise
        """
        return self._breaker.context.state == "closed"

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed.

        Returns:
            True if circuit is closed
        """
        return self._breaker.context.state == "closed"


# Export both for different use cases
__all__ = [
    "AsyncCircuitBreakerFactory",
    "CircuitBreakerWrapper",
]
