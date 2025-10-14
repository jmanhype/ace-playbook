"""
Unit tests for rate limiter (T074).

Tests verify rate limiting functionality:
- Token bucket algorithm
- Per-domain and global rate limiting
- Blocking and non-blocking modes
- Rate limiter metrics
"""

import pytest
import time
import threading
from datetime import datetime

from ace.utils.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitExceeded,
    TokenBucket,
    get_llm_rate_limiter,
    get_per_domain_rate_limiter
)


@pytest.fixture
def rate_limiter():
    """Create rate limiter with test configuration."""
    config = RateLimitConfig(
        calls_per_minute=60,
        calls_per_hour=1000,
        burst_size=10,
        strategy=RateLimitStrategy.TOKEN_BUCKET
    )
    return RateLimiter(config=config, global_limit=True)


@pytest.fixture
def per_domain_limiter():
    """Create per-domain rate limiter."""
    config = RateLimitConfig(
        calls_per_minute=30,
        calls_per_hour=500,
        burst_size=5
    )
    return RateLimiter(config=config, global_limit=False)


def test_token_bucket_initial_state():
    """T074: Token bucket should start with full capacity."""
    bucket = TokenBucket(
        capacity=10,
        tokens=10.0,
        refill_rate=1.0  # 1 token per second
    )

    assert bucket.tokens == 10.0
    assert bucket.capacity == 10
    assert bucket.consume(1) is True
    assert bucket.tokens == 9.0


def test_token_bucket_refill():
    """T074: Token bucket should refill over time."""
    bucket = TokenBucket(
        capacity=10,
        tokens=5.0,
        refill_rate=2.0,  # 2 tokens per second
        last_refill=datetime.utcnow()
    )

    # Wait 1 second
    time.sleep(1.0)

    # Should have refilled ~2 tokens
    bucket._refill()
    assert 6.5 <= bucket.tokens <= 7.5  # Allow some timing variance


def test_token_bucket_consume():
    """T074: Token bucket should consume tokens correctly."""
    bucket = TokenBucket(
        capacity=10,
        tokens=10.0,
        refill_rate=1.0
    )

    # Consume 3 tokens
    assert bucket.consume(3) is True
    assert 6.99 <= bucket.tokens <= 7.01  # Allow for tiny time variance

    # Try to consume more than available
    assert bucket.consume(10) is False
    assert 6.99 <= bucket.tokens <= 7.01  # Unchanged (within tolerance)


def test_token_bucket_get_wait_time():
    """T074: Token bucket should calculate wait time correctly."""
    bucket = TokenBucket(
        capacity=10,
        tokens=2.0,
        refill_rate=1.0  # 1 token per second
    )

    # Need 5 tokens, have 2, need 3 more
    wait_time = bucket.get_wait_time(5)
    assert 2.5 <= wait_time <= 3.5  # ~3 seconds


def test_rate_limiter_acquire_success(rate_limiter):
    """T074: Rate limiter should allow requests within limit."""
    # Should succeed immediately
    assert rate_limiter.acquire(tokens=1, block=False) is True

    # Metrics should track the call
    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["total_throttled"] == 0


def test_rate_limiter_acquire_burst(rate_limiter):
    """T074: Rate limiter should allow burst of requests."""
    # Burst size is 10, should allow 10 rapid calls
    for i in range(10):
        assert rate_limiter.acquire(tokens=1, block=False) is True

    # 11th call should be throttled
    with pytest.raises(RateLimitExceeded) as exc_info:
        rate_limiter.acquire(tokens=1, block=False)

    assert exc_info.value.retry_after > 0


def test_rate_limiter_blocking_mode(rate_limiter):
    """T074: Rate limiter should block until tokens available."""
    # Exhaust burst
    for i in range(10):
        rate_limiter.acquire(tokens=1, block=False)

    # Next call should block (but timeout quickly)
    start_time = time.time()
    with pytest.raises(RateLimitExceeded):
        rate_limiter.acquire(tokens=1, block=True, timeout=0.5)

    elapsed = time.time() - start_time
    assert 0.4 <= elapsed <= 0.6  # Should timeout around 0.5s


def test_rate_limiter_per_domain(per_domain_limiter):
    """T074: Per-domain rate limiter should isolate domains."""
    # Domain A: exhaust burst
    for i in range(5):
        assert per_domain_limiter.acquire(domain_id="domain-a", tokens=1, block=False)

    # Domain A exhausted
    with pytest.raises(RateLimitExceeded):
        per_domain_limiter.acquire(domain_id="domain-a", tokens=1, block=False)

    # Domain B should still have capacity
    assert per_domain_limiter.acquire(domain_id="domain-b", tokens=1, block=False)


def test_rate_limiter_metrics(rate_limiter):
    """T074: Rate limiter should track metrics correctly."""
    # Make some calls (within burst limit)
    for i in range(5):
        rate_limiter.acquire(tokens=1, block=False)

    # Try to exceed limit (burst is 10, so exhaust remaining 5)
    for i in range(5):
        rate_limiter.acquire(tokens=1, block=False)

    # Now try to exceed (should be throttled)
    throttled_count = 0
    for i in range(5):
        try:
            rate_limiter.acquire(tokens=1, block=False)
        except RateLimitExceeded:
            throttled_count += 1

    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == 15
    assert metrics["total_throttled"] == throttled_count
    assert throttled_count == 5


def test_rate_limiter_reset(rate_limiter):
    """T074: Rate limiter reset should restore capacity."""
    # Exhaust burst
    for i in range(10):
        rate_limiter.acquire(tokens=1, block=False)

    # Should be throttled
    with pytest.raises(RateLimitExceeded):
        rate_limiter.acquire(tokens=1, block=False)

    # Reset
    rate_limiter.reset()

    # Should work again
    assert rate_limiter.acquire(tokens=1, block=False)


def test_rate_limiter_decorator(rate_limiter):
    """T074: Rate limiter decorator should work correctly."""
    call_count = 0

    @rate_limiter
    def test_function():
        nonlocal call_count
        call_count += 1
        return "success"

    # Should allow burst calls
    for i in range(10):
        assert test_function() == "success"

    assert call_count == 10

    # Note: 11th call would block with default decorator settings
    # (not tested to avoid timeout)


def test_rate_limiter_thread_safety(rate_limiter):
    """T074: Rate limiter should be thread-safe."""
    success_count = 0
    throttled_count = 0
    lock = threading.Lock()

    def make_call():
        nonlocal success_count, throttled_count
        try:
            rate_limiter.acquire(tokens=1, block=False)
            with lock:
                success_count += 1
        except RateLimitExceeded:
            with lock:
                throttled_count += 1

    # Spawn 20 concurrent threads
    threads = []
    for i in range(20):
        t = threading.Thread(target=make_call)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Should allow ~10 (burst), throttle ~10
    assert success_count == 10
    assert throttled_count == 10


def test_get_llm_rate_limiter():
    """T074: get_llm_rate_limiter should return singleton."""
    limiter1 = get_llm_rate_limiter()
    limiter2 = get_llm_rate_limiter()

    assert limiter1 is limiter2  # Same instance


def test_get_per_domain_rate_limiter():
    """T074: get_per_domain_rate_limiter should return singleton."""
    limiter1 = get_per_domain_rate_limiter()
    limiter2 = get_per_domain_rate_limiter()

    assert limiter1 is limiter2  # Same instance


def test_rate_limiter_check_limit(rate_limiter):
    """T074: check_limit should not consume tokens."""
    # Check limit
    assert rate_limiter.check_limit(tokens=5) is True

    # Should still be able to acquire
    assert rate_limiter.acquire(tokens=5, block=False) is True

    # Now check again
    assert rate_limiter.check_limit(tokens=5) is True


def test_rate_limit_config_defaults():
    """T074: RateLimitConfig should have sensible defaults."""
    config = RateLimitConfig()

    assert config.calls_per_minute == 60
    assert config.calls_per_hour == 1000
    assert config.burst_size == 10
    assert config.strategy == RateLimitStrategy.TOKEN_BUCKET


def test_rate_limiter_refill_rate():
    """T074: Rate limiter should refill tokens at correct rate."""
    config = RateLimitConfig(
        calls_per_minute=60,  # 1 per second
        burst_size=5
    )
    limiter = RateLimiter(config=config, global_limit=True)

    # Exhaust burst
    for i in range(5):
        limiter.acquire(tokens=1, block=False)

    # Wait 2 seconds (should refill ~2 tokens)
    time.sleep(2.0)

    # Should be able to make 2 more calls
    assert limiter.acquire(tokens=1, block=False) is True
    assert limiter.acquire(tokens=1, block=False) is True

    # 3rd call should fail
    with pytest.raises(RateLimitExceeded):
        limiter.acquire(tokens=1, block=False)


def test_rate_limiter_domain_specific_reset(per_domain_limiter):
    """T074: Reset should work for specific domain."""
    # Exhaust domain A
    for i in range(5):
        per_domain_limiter.acquire(domain_id="domain-a", tokens=1, block=False)

    # Exhaust domain B
    for i in range(5):
        per_domain_limiter.acquire(domain_id="domain-b", tokens=1, block=False)

    # Reset only domain A
    per_domain_limiter.reset(domain_id="domain-a")

    # Domain A should work
    assert per_domain_limiter.acquire(domain_id="domain-a", tokens=1, block=False)

    # Domain B should still be throttled
    with pytest.raises(RateLimitExceeded):
        per_domain_limiter.acquire(domain_id="domain-b", tokens=1, block=False)
