"""
Unit tests for circuit breaker pattern implementation (T071).

Tests verify circuit breaker behavior:
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Failure threshold triggering
- Recovery timeout behavior
- Metrics collection
- Decorator pattern usage
- Thread safety
"""

import pytest
import time
from unittest.mock import Mock

from ace.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    circuit,
    get_circuit_breaker,
    register_circuit_breaker,
    get_all_circuit_breakers,
    get_all_metrics
)


class TestException(Exception):
    """Test exception for circuit breaker tests."""
    pass


def test_circuit_breaker_initial_state():
    """T071: Circuit breaker should start in CLOSED state."""
    breaker = CircuitBreaker(name="test-initial")

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_successful_call():
    """T071: Successful calls should keep circuit CLOSED."""
    breaker = CircuitBreaker(name="test-success")

    def successful_func():
        return "success"

    result = breaker.call(successful_func)

    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_failure_increments_count():
    """T071: Failures should increment failure count."""
    breaker = CircuitBreaker(
        failure_threshold=3,
        expected_exception=TestException,
        name="test-failure-count"
    )

    def failing_func():
        raise TestException("Test failure")

    # First failure
    with pytest.raises(TestException):
        breaker.call(failing_func)

    assert breaker.failure_count == 1
    assert breaker.state == CircuitState.CLOSED

    # Second failure
    with pytest.raises(TestException):
        breaker.call(failing_func)

    assert breaker.failure_count == 2
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_opens_after_threshold():
    """T071: Circuit should open after failure threshold is reached."""
    breaker = CircuitBreaker(
        failure_threshold=3,
        expected_exception=TestException,
        name="test-open-threshold"
    )

    def failing_func():
        raise TestException("Test failure")

    # Trigger failures up to threshold
    for i in range(3):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    # Circuit should now be OPEN
    assert breaker.state == CircuitState.OPEN
    assert breaker.failure_count == 3


def test_circuit_breaker_rejects_when_open():
    """T071: Open circuit should reject calls immediately with CircuitBreakerError."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=60,
        expected_exception=TestException,
        name="test-reject-open"
    )

    def failing_func():
        raise TestException("Test failure")

    # Open the circuit
    for i in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.state == CircuitState.OPEN

    # Next call should be rejected immediately
    with pytest.raises(CircuitBreakerError, match="Circuit breaker.*is OPEN"):
        breaker.call(failing_func)


def test_circuit_breaker_half_open_after_timeout():
    """T071: Circuit should enter HALF_OPEN state after recovery timeout."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=1,  # 1 second timeout
        expected_exception=TestException,
        name="test-half-open"
    )

    def failing_func():
        raise TestException("Test failure")

    # Open the circuit
    for i in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(1.1)

    # Next call should transition to HALF_OPEN
    with pytest.raises(TestException):
        breaker.call(failing_func)

    # Circuit should have been HALF_OPEN during the call
    # (now back to OPEN due to failure)
    assert breaker.state == CircuitState.OPEN


def test_circuit_breaker_closes_after_successful_half_open():
    """T071: Circuit should close after successful call in HALF_OPEN state."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=1,
        expected_exception=TestException,
        name="test-close-after-half-open"
    )

    def failing_func():
        raise TestException("Test failure")

    def successful_func():
        return "recovered"

    # Open the circuit
    for i in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(1.1)

    # Successful call should close circuit
    result = breaker.call(successful_func)

    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_resets_failure_count_on_success():
    """T071: Successful call should reset failure count."""
    breaker = CircuitBreaker(
        failure_threshold=5,
        expected_exception=TestException,
        name="test-reset-count"
    )

    def failing_func():
        raise TestException("Test failure")

    def successful_func():
        return "success"

    # Accumulate 3 failures
    for i in range(3):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.failure_count == 3

    # Successful call resets count
    result = breaker.call(successful_func)

    assert result == "success"
    assert breaker.failure_count == 0


def test_circuit_breaker_metrics():
    """T071: Circuit breaker should track metrics accurately."""
    breaker = CircuitBreaker(
        failure_threshold=3,
        expected_exception=TestException,
        name="test-metrics"
    )

    def failing_func():
        raise TestException("Test failure")

    def successful_func():
        return "success"

    # 2 successes
    breaker.call(successful_func)
    breaker.call(successful_func)

    # 3 failures (opens circuit)
    for i in range(3):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    # Get metrics
    metrics = breaker.get_metrics()

    assert metrics["name"] == "test-metrics"
    assert metrics["state"] == CircuitState.OPEN.value
    assert metrics["total_calls"] == 5
    assert metrics["total_successes"] == 2
    assert metrics["total_failures"] == 3
    assert metrics["failure_count"] == 3
    assert metrics["open_count"] == 1
    assert metrics["last_failure_time"] is not None


def test_circuit_breaker_decorator():
    """T071: @circuit decorator should apply circuit breaker to function."""
    call_count = [0]

    @circuit(
        failure_threshold=2,
        recovery_timeout=60,
        expected_exception=TestException,
        name="test-decorator"
    )
    def decorated_func(should_fail: bool):
        call_count[0] += 1
        if should_fail:
            raise TestException("Decorated failure")
        return "success"

    # Successful call
    result = decorated_func(False)
    assert result == "success"
    assert call_count[0] == 1

    # Two failures (opens circuit)
    for i in range(2):
        with pytest.raises(TestException):
            decorated_func(True)

    assert call_count[0] == 3

    # Circuit should be open - next call rejected without executing function
    with pytest.raises(CircuitBreakerError):
        decorated_func(False)

    # Function was not called (count unchanged)
    assert call_count[0] == 3


def test_circuit_breaker_decorator_has_breaker_attribute():
    """T071: Decorated function should expose circuit breaker for metrics."""
    @circuit(name="test-expose-breaker")
    def my_func():
        return "test"

    # Decorator should attach breaker instance
    assert hasattr(my_func, "_circuit_breaker")
    assert isinstance(my_func._circuit_breaker, CircuitBreaker)
    assert my_func._circuit_breaker.name == "test-expose-breaker"


def test_circuit_breaker_global_registry():
    """T071: Circuit breakers should be registerable in global registry."""
    breaker1 = CircuitBreaker(name="registry-test-1")
    breaker2 = CircuitBreaker(name="registry-test-2")

    register_circuit_breaker(breaker1)
    register_circuit_breaker(breaker2)

    # Retrieve from registry
    retrieved1 = get_circuit_breaker("registry-test-1")
    retrieved2 = get_circuit_breaker("registry-test-2")

    assert retrieved1 is breaker1
    assert retrieved2 is breaker2

    # Get all breakers
    all_breakers = get_all_circuit_breakers()
    assert "registry-test-1" in all_breakers
    assert "registry-test-2" in all_breakers


def test_circuit_breaker_get_all_metrics():
    """T071: get_all_metrics should return metrics from all registered breakers."""
    # Register some breakers with activity
    breaker1 = CircuitBreaker(name="metrics-test-1")
    breaker2 = CircuitBreaker(name="metrics-test-2")

    register_circuit_breaker(breaker1)
    register_circuit_breaker(breaker2)

    # Generate some activity
    breaker1.call(lambda: "success")
    breaker2.call(lambda: "success")

    # Get all metrics
    all_metrics = get_all_metrics()

    # Should contain metrics from both breakers
    names = {m["name"] for m in all_metrics}
    assert "metrics-test-1" in names
    assert "metrics-test-2" in names


def test_circuit_breaker_different_exception_types():
    """T071: Circuit breaker should only catch expected exception type."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        expected_exception=TestException,
        name="test-exception-type"
    )

    def func_with_unexpected_exception():
        raise ValueError("Unexpected exception type")

    # Unexpected exception should not be caught by circuit breaker
    with pytest.raises(ValueError):
        breaker.call(func_with_unexpected_exception)

    # Failure count should NOT increment for unexpected exceptions
    assert breaker.failure_count == 0
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_with_args_and_kwargs():
    """T071: Circuit breaker should pass through args and kwargs."""
    breaker = CircuitBreaker(name="test-args-kwargs")

    def func_with_params(a, b, c=None):
        return f"a={a}, b={b}, c={c}"

    result = breaker.call(func_with_params, "arg1", "arg2", c="kwarg1")

    assert result == "a=arg1, b=arg2, c=kwarg1"


def test_circuit_breaker_open_count_increments():
    """T071: open_count should increment each time circuit opens."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=1,
        expected_exception=TestException,
        name="test-open-count"
    )

    def failing_func():
        raise TestException("Failure")

    def successful_func():
        return "success"

    # First circuit opening
    for i in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.state == CircuitState.OPEN
    metrics1 = breaker.get_metrics()
    assert metrics1["open_count"] == 1

    # Wait and recover
    time.sleep(1.1)
    breaker.call(successful_func)

    assert breaker.state == CircuitState.CLOSED

    # Second circuit opening
    for i in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_func)

    assert breaker.state == CircuitState.OPEN
    metrics2 = breaker.get_metrics()
    assert metrics2["open_count"] == 2


def test_circuit_breaker_thread_safety():
    """T071: Circuit breaker should be thread-safe for concurrent access."""
    import threading

    breaker = CircuitBreaker(
        failure_threshold=10,
        name="test-thread-safety"
    )

    results = []

    def worker():
        try:
            result = breaker.call(lambda: "success")
            results.append(result)
        except Exception as e:
            results.append(str(e))

    # Launch 100 concurrent calls
    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All calls should succeed
    assert len(results) == 100
    assert all(r == "success" for r in results)

    # Metrics should be accurate
    metrics = breaker.get_metrics()
    assert metrics["total_calls"] == 100
    assert metrics["total_successes"] == 100
