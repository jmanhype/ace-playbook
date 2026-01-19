"""Common patterns for BLACKICE 3.0.

This module provides functional programming patterns:
- Result: Success/failure container with error handling
- Either: Left/Right discriminated union
- retry: Decorator for retrying failed operations
- circuit_breaker: Pattern for failing fast on repeated errors
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, ParamSpec, TypeVar

from blackice.primitives.errors import BlackiceError

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
L = TypeVar("L")
R = TypeVar("R")
P = ParamSpec("P")


# === Result Pattern ===


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful result container."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Get the success value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the success value or a default."""
        return self.value

    def map(self, fn: Callable[[T], R]) -> Result[R, E]:
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], R]) -> Result[T, R]:
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Failed result container."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        """Raise the contained error."""
        raise self.error

    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default

    def map(self, fn: Callable[[T], R]) -> Result[R, E]:
        """Transform the success value (no-op for Err)."""
        return self  # type: ignore

    def map_err(self, fn: Callable[[E], R]) -> Result[T, R]:
        """Transform the error."""
        return Err(fn(self.error))


Result = Ok[T] | Err[E]


def ok(value: T) -> Ok[T]:
    """Create a successful result."""
    return Ok(value)


def err(error: E) -> Err[E]:
    """Create a failed result."""
    return Err(error)


# === Either Pattern ===


@dataclass(frozen=True, slots=True)
class Left(Generic[L]):
    """Left side of Either."""

    value: L

    def is_left(self) -> bool:
        return True

    def is_right(self) -> bool:
        return False

    def map_left(self, fn: Callable[[L], T]) -> Either[T, R]:
        """Transform the left value."""
        return Left(fn(self.value))

    def map_right(self, fn: Callable[[R], T]) -> Either[L, T]:
        """Transform the right value (no-op for Left)."""
        return self  # type: ignore


@dataclass(frozen=True, slots=True)
class Right(Generic[R]):
    """Right side of Either."""

    value: R

    def is_left(self) -> bool:
        return False

    def is_right(self) -> bool:
        return True

    def map_left(self, fn: Callable[[L], T]) -> Either[T, R]:
        """Transform the left value (no-op for Right)."""
        return self  # type: ignore

    def map_right(self, fn: Callable[[R], T]) -> Either[L, T]:
        """Transform the right value."""
        return Right(fn(self.value))


Either = Left[L] | Right[R]


def left(value: L) -> Left[L]:
    """Create a Left value."""
    return Left(value)


def right(value: R) -> Right[R]:
    """Create a Right value."""
    return Right(value)


# === Retry Pattern ===


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retrying failed synchronous operations.

    Args:
        max_attempts: Maximum number of attempts (including initial)
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt)

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data():
            return requests.get(url).json()
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(current_delay)
                        current_delay *= backoff

            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], Awaitable[None] | None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for retrying failed async operations.

    Args:
        max_attempts: Maximum number of attempts (including initial)
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt)

    Example:
        @async_retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get(url)
    """

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if on_retry:
                            result = on_retry(e, attempt)
                            if asyncio.iscoroutine(result):
                                await result
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


# === Circuit Breaker Pattern ===


class CircuitBreaker:
    """Circuit breaker for failing fast on repeated errors.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests fail immediately
    - HALF_OPEN: Testing if service has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        @breaker
        async def call_service():
            return await http_client.get(url)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        name: str = "default",
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Successes needed in half-open to close
            name: Name for logging/identification
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._state == self.OPEN:
            if self._should_attempt_reset():
                self._state = self.HALF_OPEN
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == self.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = self.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self._success_count = 0
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN

    def __call__(
        self, fn: Callable[P, Awaitable[T]]
    ) -> Callable[P, Awaitable[T]]:
        """Decorator to wrap async function with circuit breaker."""

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.state == self.OPEN:
                raise BlackiceError(
                    f"Circuit breaker '{self.name}' is open",
                    code="E7001",
                    recoverable=True,
                    context={"circuit": self.name, "state": self._state},
                )

            try:
                result = await fn(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
