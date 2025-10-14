"""
Rate Limiter for LLM API Calls

Implements token bucket algorithm with per-domain rate limiting to prevent:
- DoS attacks
- LLM API quota exhaustion
- Cost overruns

T074: Add rate limiting for LLM API calls (Engineering Review recommendation)
"""

import time
import threading
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="rate_limiter")


class RateLimitStrategy(Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"      # Allows bursts, smooth long-term
    SLIDING_WINDOW = "sliding_window"  # Precise limits, no bursts


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    calls_per_minute: int = 60          # Maximum calls per minute
    calls_per_hour: int = 1000          # Maximum calls per hour
    burst_size: int = 10                # Allow burst of N calls
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int                        # Maximum tokens
    tokens: float                        # Current tokens
    refill_rate: float                   # Tokens per second
    last_refill: datetime = field(default_factory=datetime.utcnow)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get seconds to wait before tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if tokens available now)
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                return 0.0

            # Calculate time needed to accumulate tokens
            tokens_needed = tokens - self.tokens
            wait_seconds = tokens_needed / self.refill_rate
            return wait_seconds


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: float):
        self.retry_after = retry_after
        super().__init__(message)


class RateLimiter:
    """
    Rate limiter for LLM API calls using token bucket algorithm.

    Implements per-domain rate limiting with configurable limits.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        global_limit: bool = False
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration (defaults to 60/min, 1000/hour)
            global_limit: If True, apply limit globally; if False, per-domain
        """
        self.config = config or RateLimitConfig()
        self.global_limit = global_limit

        # Per-domain token buckets (or single global bucket)
        self._buckets: Dict[str, TokenBucket] = {}
        self._buckets_lock = threading.Lock()

        # Metrics
        self._total_calls = 0
        self._total_throttled = 0
        self._metrics_lock = threading.Lock()

        logger.info(
            "rate_limiter_initialized",
            calls_per_minute=self.config.calls_per_minute,
            calls_per_hour=self.config.calls_per_hour,
            burst_size=self.config.burst_size,
            global_limit=global_limit
        )

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """
        Get or create token bucket for domain/global key.

        Args:
            key: Domain ID or "global" for global limit

        Returns:
            TokenBucket instance
        """
        with self._buckets_lock:
            if key not in self._buckets:
                # Create bucket with per-minute refill rate
                refill_rate = self.config.calls_per_minute / 60.0
                bucket = TokenBucket(
                    capacity=self.config.burst_size,
                    tokens=float(self.config.burst_size),
                    refill_rate=refill_rate
                )
                self._buckets[key] = bucket
                logger.debug(
                    "token_bucket_created",
                    key=key,
                    capacity=self.config.burst_size,
                    refill_rate=refill_rate
                )

            return self._buckets[key]

    def check_limit(self, domain_id: Optional[str] = None, tokens: int = 1) -> bool:
        """
        Check if request is within rate limit without consuming tokens.

        Args:
            domain_id: Domain ID (required if global_limit=False)
            tokens: Number of tokens to check for

        Returns:
            True if request would be allowed
        """
        key = "global" if self.global_limit else (domain_id or "default")
        bucket = self._get_or_create_bucket(key)

        with bucket._lock:
            bucket._refill()
            return bucket.tokens >= tokens

    def acquire(
        self,
        domain_id: Optional[str] = None,
        tokens: int = 1,
        block: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire tokens from rate limiter.

        Args:
            domain_id: Domain ID (required if global_limit=False)
            tokens: Number of tokens to acquire (default: 1)
            block: If True, block until tokens available
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if limit exceeded (non-blocking only)

        Raises:
            RateLimitExceeded: If timeout expires or non-blocking and limit exceeded
        """
        key = "global" if self.global_limit else (domain_id or "default")
        bucket = self._get_or_create_bucket(key)

        # Track metrics
        with self._metrics_lock:
            self._total_calls += 1

        # Try to consume tokens
        if bucket.consume(tokens):
            logger.debug("rate_limit_acquired", key=key, tokens=tokens)
            return True

        # Rate limit exceeded
        with self._metrics_lock:
            self._total_throttled += 1

        wait_time = bucket.get_wait_time(tokens)

        logger.warning(
            "rate_limit_exceeded",
            key=key,
            tokens=tokens,
            wait_time_seconds=wait_time
        )

        if not block:
            raise RateLimitExceeded(
                f"Rate limit exceeded for '{key}'. Retry after {wait_time:.2f}s",
                retry_after=wait_time
            )

        # Block until tokens available (with timeout)
        start_time = time.time()
        while True:
            if timeout and (time.time() - start_time) >= timeout:
                raise RateLimitExceeded(
                    f"Rate limit timeout for '{key}' after {timeout}s",
                    retry_after=wait_time
                )

            # Calculate sleep time
            sleep_duration = min(wait_time, 0.1)  # Sleep in 100ms increments
            time.sleep(sleep_duration)

            # Try again
            if bucket.consume(tokens):
                logger.info(
                    "rate_limit_acquired_after_wait",
                    key=key,
                    tokens=tokens,
                    wait_time_seconds=time.time() - start_time
                )
                return True

            # Recalculate wait time
            wait_time = bucket.get_wait_time(tokens)

    def __call__(
        self,
        func: Optional[Callable] = None,
        domain_id: Optional[str] = None,
        tokens: int = 1
    ) -> Any:
        """
        Rate-limited function decorator.

        Usage:
            @rate_limiter
            def my_function():
                pass

            # Or with parameters:
            @rate_limiter(domain_id="customer-a", tokens=2)
            def expensive_function():
                pass

        Args:
            func: Function to wrap
            domain_id: Domain ID for rate limiting
            tokens: Tokens to consume per call

        Returns:
            Wrapped function
        """
        def decorator(f: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract domain_id from kwargs if not provided
                call_domain_id = domain_id or kwargs.get("domain_id")

                # Acquire rate limit
                self.acquire(domain_id=call_domain_id, tokens=tokens, block=True)

                # Call function
                return f(*args, **kwargs)

            wrapper.__name__ = f.__name__
            wrapper.__doc__ = f.__doc__
            return wrapper

        # Support both @rate_limiter and @rate_limiter(...)
        if func is not None:
            return decorator(func)
        return decorator

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get rate limiter metrics.

        Returns:
            Dictionary with metrics
        """
        with self._metrics_lock:
            metrics = {
                "total_calls": self._total_calls,
                "total_throttled": self._total_throttled,
                "throttle_rate": (
                    self._total_throttled / self._total_calls
                    if self._total_calls > 0 else 0.0
                ),
                "buckets": {}
            }

        # Add per-bucket metrics
        with self._buckets_lock:
            for key, bucket in self._buckets.items():
                with bucket._lock:
                    bucket._refill()
                    metrics["buckets"][key] = {
                        "tokens_available": bucket.tokens,
                        "capacity": bucket.capacity,
                        "refill_rate": bucket.refill_rate
                    }

        return metrics

    def reset(self, domain_id: Optional[str] = None) -> None:
        """
        Reset rate limiter state.

        Args:
            domain_id: Domain ID to reset (None = reset all)
        """
        with self._buckets_lock:
            if domain_id:
                key = "global" if self.global_limit else domain_id
                if key in self._buckets:
                    bucket = self._buckets[key]
                    with bucket._lock:
                        bucket.tokens = float(bucket.capacity)
                        bucket.last_refill = datetime.utcnow()
                    logger.info("rate_limiter_reset", key=key)
            else:
                # Reset all buckets
                for key, bucket in self._buckets.items():
                    with bucket._lock:
                        bucket.tokens = float(bucket.capacity)
                        bucket.last_refill = datetime.utcnow()
                logger.info("rate_limiter_reset_all")

        # Reset metrics
        with self._metrics_lock:
            self._total_calls = 0
            self._total_throttled = 0


# Global rate limiters
_llm_rate_limiter: Optional[RateLimiter] = None
_per_domain_rate_limiter: Optional[RateLimiter] = None


def get_llm_rate_limiter(
    calls_per_minute: int = 60,
    calls_per_hour: int = 1000,
    burst_size: int = 10
) -> RateLimiter:
    """
    Get global LLM rate limiter (singleton).

    Args:
        calls_per_minute: Maximum LLM calls per minute
        calls_per_hour: Maximum LLM calls per hour
        burst_size: Allow burst of N calls

    Returns:
        Global RateLimiter instance
    """
    global _llm_rate_limiter
    if _llm_rate_limiter is None:
        config = RateLimitConfig(
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            burst_size=burst_size
        )
        _llm_rate_limiter = RateLimiter(config=config, global_limit=True)

    return _llm_rate_limiter


def get_per_domain_rate_limiter(
    calls_per_minute: int = 30,
    calls_per_hour: int = 500,
    burst_size: int = 5
) -> RateLimiter:
    """
    Get per-domain rate limiter (singleton).

    Args:
        calls_per_minute: Maximum calls per minute per domain
        calls_per_hour: Maximum calls per hour per domain
        burst_size: Allow burst of N calls per domain

    Returns:
        Per-domain RateLimiter instance
    """
    global _per_domain_rate_limiter
    if _per_domain_rate_limiter is None:
        config = RateLimitConfig(
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            burst_size=burst_size
        )
        _per_domain_rate_limiter = RateLimiter(config=config, global_limit=False)

    return _per_domain_rate_limiter
