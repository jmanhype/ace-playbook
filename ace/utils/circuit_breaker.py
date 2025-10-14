"""
Circuit Breaker Pattern for External API Calls (T071).

Implements circuit breaker to prevent cascading failures when external
services (OpenAI, Anthropic) are down or experiencing issues.

Pattern States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, reject requests immediately
- HALF_OPEN: Testing if service recovered, allow limited requests

Based on Martin Fowler's Circuit Breaker pattern:
https://martinfowler.com/bliki/CircuitBreaker.html
"""

import time
from typing import Callable, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import threading
from functools import wraps

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Rejecting requests (failure threshold exceeded)
    HALF_OPEN = "half_open"    # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and rejects a request."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for external API calls with automatic recovery testing.

    Prevents cascading failures by failing fast when a service is down,
    then automatically testing recovery after a timeout period.

    Attributes:
        failure_threshold: Number of consecutive failures before opening (default: 5)
        recovery_timeout: Seconds to wait before testing recovery (default: 60)
        expected_exception: Exception type to count as failure (default: Exception)
        name: Circuit breaker name for logging
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds before testing recovery (HALF_OPEN state)
            expected_exception: Exception type that counts as failure
            name: Circuit breaker identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._open_count = 0  # Number of times circuit opened

        logger.info(
            "circuit_breaker_initialized",
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current consecutive failure count."""
        return self._failure_count

    def get_metrics(self) -> dict:
        """
        Get circuit breaker metrics.

        Returns:
            Dict with metrics: total_calls, failures, successes, state, etc.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "open_count": self._open_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            }

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery test."""
        if self._last_failure_time is None:
            return False

        elapsed = datetime.now() - self._last_failure_time
        return elapsed.total_seconds() >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._total_successes += 1
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                # Recovery test passed - close circuit
                logger.info(
                    "circuit_breaker_closed",
                    name=self.name,
                    recovery_successful=True
                )
                self._state = CircuitState.CLOSED

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
                error=str(exception),
                error_type=type(exception).__name__
            )

            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._open_count += 1
                    logger.error(
                        "circuit_breaker_opened",
                        name=self.name,
                        failure_count=self._failure_count,
                        threshold=self.failure_threshold,
                        open_count=self._open_count
                    )
                self._state = CircuitState.OPEN

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result if circuit is closed/half-open

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function raises (and failure is recorded)
        """
        with self._lock:
            self._total_calls += 1

            # Check state and decide whether to allow call
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Transition to HALF_OPEN for recovery test
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        elapsed_seconds=(datetime.now() - self._last_failure_time).total_seconds()
                    )
                    self._state = CircuitState.HALF_OPEN
                else:
                    # Circuit still open - reject immediately
                    logger.debug(
                        "circuit_breaker_rejected",
                        name=self.name,
                        state=self._state.value
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable after {self._failure_count} failures. "
                        f"Retry after {self.recovery_timeout}s."
                    )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure(e)
            raise


def circuit(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    """
    Decorator to apply circuit breaker pattern to a function.

    Args:
        failure_threshold: Consecutive failures before opening circuit
        recovery_timeout: Seconds before testing recovery
        expected_exception: Exception type that triggers circuit breaker
        name: Circuit breaker name (defaults to function name)

    Example:
        @circuit(failure_threshold=3, recovery_timeout=30, name="openai")
        def call_openai_api(prompt: str) -> str:
            return openai.ChatCompletion.create(...)
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        # Attach breaker instance to wrapper for metrics access
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


# Global registry for circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """
    Get circuit breaker by name from global registry.

    Args:
        name: Circuit breaker name

    Returns:
        CircuitBreaker instance or None if not found
    """
    with _registry_lock:
        return _circuit_breakers.get(name)


def register_circuit_breaker(breaker: CircuitBreaker) -> None:
    """
    Register circuit breaker in global registry.

    Args:
        breaker: CircuitBreaker instance to register
    """
    with _registry_lock:
        _circuit_breakers[breaker.name] = breaker
        logger.debug("circuit_breaker_registered", name=breaker.name)


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """
    Get all registered circuit breakers.

    Returns:
        Dict mapping name to CircuitBreaker instance
    """
    with _registry_lock:
        return dict(_circuit_breakers)


def get_all_metrics() -> list[dict]:
    """
    Get metrics from all registered circuit breakers.

    Returns:
        List of metric dicts, one per circuit breaker
    """
    breakers = get_all_circuit_breakers()
    return [breaker.get_metrics() for breaker in breakers.values()]
