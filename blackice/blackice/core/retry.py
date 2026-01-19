"""Retry logic with exponential backoff for BLACKICE 3.0.

Provides configurable retry behavior for resilient execution
of operations that may fail transiently.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from blackice.primitives.errors import BlackiceError

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Exceptions to retry on (None = all BlackiceError with recoverable=True)
    retryable_exceptions: tuple[type[Exception], ...] | None = None

    # Callback for retry events
    on_retry: Callable[[Exception, int, float], Awaitable[None] | None] | None = None


@dataclass
class RetryStats:
    """Statistics from a retry operation."""

    attempts: int = 0
    total_delay: float = 0.0
    last_exception: Exception | None = None
    success: bool = False


class RetryExhaustedError(BlackiceError):
    """All retry attempts exhausted."""

    code = "E7001"

    def __init__(
        self,
        operation: str,
        attempts: int,
        last_exception: Exception,
    ) -> None:
        super().__init__(
            f"Retry exhausted after {attempts} attempts for '{operation}'",
            context={
                "operation": operation,
                "attempts": attempts,
                "last_error": str(last_exception),
            },
        )
        self.__cause__ = last_exception


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for next retry attempt.

    Uses exponential backoff with optional jitter.
    """
    delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """Determine if an exception is retryable."""
    # Explicit retryable exceptions
    if config.retryable_exceptions:
        return isinstance(exception, config.retryable_exceptions)

    # Default: retry BlackiceError with recoverable=True
    if isinstance(exception, BlackiceError):
        return exception.recoverable

    # Don't retry unknown exceptions by default
    return False


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    operation_name: str = "operation",
) -> tuple[T, RetryStats]:
    """Execute an async operation with retry logic.

    Args:
        operation: Async callable to execute
        config: Retry configuration
        operation_name: Name for error messages

    Returns:
        Tuple of (result, stats)

    Raises:
        RetryExhaustedError: If all attempts fail
    """
    if config is None:
        config = RetryConfig()

    stats = RetryStats()

    for attempt in range(1, config.max_attempts + 1):
        stats.attempts = attempt

        try:
            result = await operation()
            stats.success = True
            return result, stats

        except Exception as e:
            stats.last_exception = e

            # Check if we should retry
            if attempt < config.max_attempts and should_retry(e, config):
                delay = calculate_delay(attempt, config)
                stats.total_delay += delay

                # Call retry callback if provided
                if config.on_retry:
                    callback_result = config.on_retry(e, attempt, delay)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result

                await asyncio.sleep(delay)
            else:
                # No more retries
                raise RetryExhaustedError(
                    operation_name,
                    stats.attempts,
                    e,
                ) from e

    # Should never reach here, but satisfy type checker
    assert stats.last_exception is not None
    raise RetryExhaustedError(
        operation_name,
        stats.attempts,
        stats.last_exception,
    )


def retry_sync(
    operation: Callable[[], T],
    config: RetryConfig | None = None,
    operation_name: str = "operation",
) -> tuple[T, RetryStats]:
    """Execute a sync operation with retry logic.

    Args:
        operation: Callable to execute
        config: Retry configuration
        operation_name: Name for error messages

    Returns:
        Tuple of (result, stats)

    Raises:
        RetryExhaustedError: If all attempts fail
    """
    import time

    if config is None:
        config = RetryConfig()

    stats = RetryStats()

    for attempt in range(1, config.max_attempts + 1):
        stats.attempts = attempt

        try:
            result = operation()
            stats.success = True
            return result, stats

        except Exception as e:
            stats.last_exception = e

            # Check if we should retry
            if attempt < config.max_attempts and should_retry(e, config):
                delay = calculate_delay(attempt, config)
                stats.total_delay += delay

                # Call retry callback if provided (sync only)
                if config.on_retry:
                    callback_result = config.on_retry(e, attempt, delay)
                    if asyncio.iscoroutine(callback_result):
                        raise ValueError("Sync retry cannot use async callbacks")

                time.sleep(delay)
            else:
                # No more retries
                raise RetryExhaustedError(
                    operation_name,
                    stats.attempts,
                    e,
                ) from e

    # Should never reach here
    assert stats.last_exception is not None
    raise RetryExhaustedError(
        operation_name,
        stats.attempts,
        stats.last_exception,
    )


class RetryContext:
    """Context manager for manual retry control.

    Example:
        async with RetryContext(config) as ctx:
            while ctx.should_continue:
                try:
                    result = await operation()
                    break
                except Exception as e:
                    await ctx.handle_error(e)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()
        self.stats = RetryStats()
        self._exhausted = False

    @property
    def should_continue(self) -> bool:
        """Check if retry loop should continue."""
        return not self._exhausted and self.stats.attempts < self.config.max_attempts

    @property
    def current_attempt(self) -> int:
        """Get current attempt number (1-based)."""
        return self.stats.attempts + 1

    async def handle_error(self, exception: Exception) -> None:
        """Handle an error during retry loop.

        Raises RetryExhaustedError if no more retries.
        """
        self.stats.last_exception = exception
        self.stats.attempts += 1

        if not should_retry(exception, self.config):
            self._exhausted = True
            raise

        if self.stats.attempts >= self.config.max_attempts:
            self._exhausted = True
            raise RetryExhaustedError(
                "operation",
                self.stats.attempts,
                exception,
            ) from exception

        delay = calculate_delay(self.stats.attempts, self.config)
        self.stats.total_delay += delay

        if self.config.on_retry:
            callback_result = self.config.on_retry(
                exception, self.stats.attempts, delay
            )
            if asyncio.iscoroutine(callback_result):
                await callback_result

        await asyncio.sleep(delay)

    async def __aenter__(self) -> RetryContext:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
