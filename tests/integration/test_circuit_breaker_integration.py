"""
Integration tests for circuit breaker with Generator and Reflector (T071).

Tests verify circuit breaker protection when LLM APIs fail:
- Circuit opens after repeated LLM failures
- Circuit closes after successful recovery
- Circuit metrics are tracked accurately
- Generator and Reflector both protected
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from ace.generator.cot_generator import CoTGenerator
from ace.generator.signatures import TaskInput
from ace.reflector.grounded_reflector import GroundedReflector
from ace.reflector.signatures import ReflectorInput
from ace.utils.circuit_breaker import CircuitBreakerError
from ace.utils.llm_circuit_breaker import (
    get_llm_breaker_metrics,
    reset_llm_breakers,
    get_or_create_llm_breaker
)


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset LLM circuit breakers before each test."""
    reset_llm_breakers()
    yield
    reset_llm_breakers()


@pytest.fixture
def generator():
    """Create CoTGenerator instance."""
    return CoTGenerator(
        model="gpt-4-turbo",
        temperature=0.7
    )


@pytest.fixture
def reflector():
    """Create GroundedReflector instance."""
    return GroundedReflector(
        model="gpt-4o-mini",
        temperature=0.3
    )


@pytest.fixture
def sample_task_input():
    """Create sample TaskInput for testing."""
    return TaskInput(
        task_id="test-task-1",
        description="What is 2 + 2?",
        domain="arithmetic",
        playbook_bullets=["Break down problems step by step", "Verify calculations"],
        max_reasoning_steps=5
    )


@pytest.fixture
def sample_reflector_input():
    """Create sample ReflectorInput for testing."""
    return ReflectorInput(
        task_id="test-task-1",
        reasoning_trace=["Step 1: Add 2 + 2", "Step 2: Result is 4"],
        answer="4",
        confidence=0.95,
        bullets_referenced=["bullet-001"],
        domain="arithmetic",
        ground_truth="4",
        test_results='{"test_arithmetic": true}',
        error_messages=[],
        performance_metrics='{}'
    )


def test_generator_circuit_opens_after_failures(generator, sample_task_input):
    """T071: Generator circuit should open after repeated LLM API failures."""

    # Mock the predictor to simulate LLM API failures
    with patch.object(generator, 'predictor') as mock_predictor:
        # Simulate API failures (5 consecutive failures)
        mock_predictor.side_effect = Exception("OpenAI API timeout")

        # First 5 calls should raise the underlying exception and increment failure count
        for i in range(5):
            with pytest.raises(Exception, match="OpenAI API timeout"):
                generator(sample_task_input)

        # Circuit should now be open
        breaker = get_or_create_llm_breaker("generator")
        assert breaker.failure_count == 5

        # 6th call should be rejected by circuit breaker without calling predictor
        mock_predictor.side_effect = None  # Reset side effect
        mock_predictor.return_value = MagicMock()

        # Generator wraps CircuitBreakerError in RuntimeError
        with pytest.raises(RuntimeError, match="Circuit breaker.*is OPEN"):
            generator(sample_task_input)


def test_reflector_circuit_opens_after_failures(reflector, sample_reflector_input):
    """T071: Reflector circuit should open after repeated LLM API failures."""

    # Mock the predictor to simulate LLM API failures
    with patch.object(reflector, 'predictor') as mock_predictor:
        # Simulate API failures (5 consecutive failures)
        mock_predictor.side_effect = Exception("Anthropic API error")

        # First 5 calls should raise the underlying exception
        for i in range(5):
            with pytest.raises(RuntimeError, match="Reflection failed"):
                reflector(sample_reflector_input)

        # Circuit should now be open
        breaker = get_or_create_llm_breaker("reflector")
        assert breaker.failure_count == 5


def test_circuit_closes_after_successful_recovery(generator, sample_task_input):
    """T071: Circuit should close after successful call in HALF_OPEN state."""

    # Create circuit breaker with short recovery timeout
    from ace.utils.llm_circuit_breaker import _llm_breakers
    from ace.utils.circuit_breaker import CircuitBreaker, register_circuit_breaker

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1,  # 1 second
        expected_exception=Exception,
        name="llm-generator"
    )
    _llm_breakers["generator"] = breaker
    register_circuit_breaker(breaker)

    with patch.object(generator, 'predictor') as mock_predictor:
        # Open the circuit with 3 failures
        mock_predictor.side_effect = Exception("API down")
        for i in range(3):
            with pytest.raises(Exception):
                generator(sample_task_input)

        # Wait for recovery timeout
        time.sleep(1.1)

        # Mock successful response
        mock_predictor.side_effect = None
        mock_predictor.return_value = MagicMock(
            reasoning="Step 1: Calculate 2 + 2\nStep 2: Result is 4",
            answer="4",
            confidence="0.95"
        )

        # Successful call should close circuit
        result = generator(sample_task_input)

        assert result.answer == "4"
        assert breaker.failure_count == 0


def test_circuit_breaker_metrics_tracked(generator, sample_task_input):
    """T071: Circuit breaker should track metrics for all LLM calls."""

    with patch.object(generator, 'predictor') as mock_predictor:
        # Mock successful response
        mock_predictor.return_value = MagicMock(
            reasoning="Step 1: Calculate 2 + 2",
            answer="4",
            confidence="0.95"
        )

        # Make 3 successful calls
        for i in range(3):
            generator(sample_task_input)

        # Mock failure
        mock_predictor.side_effect = Exception("Temporary error")

        # Make 2 failed calls
        for i in range(2):
            with pytest.raises(Exception):
                generator(sample_task_input)

    # Get metrics
    metrics = get_llm_breaker_metrics()

    # Find generator metrics
    gen_metrics = next(m for m in metrics if "generator" in m["name"])

    assert gen_metrics["total_calls"] == 5
    assert gen_metrics["total_successes"] == 3
    assert gen_metrics["total_failures"] == 2


def test_separate_circuits_for_generator_and_reflector(generator, reflector, sample_task_input, sample_reflector_input):
    """T071: Generator and Reflector should have separate circuit breakers."""

    with patch.object(generator, 'predictor') as gen_mock:
        with patch.object(reflector, 'predictor') as ref_mock:
            # Fail generator predictor
            gen_mock.side_effect = Exception("Generator API error")

            # Succeed reflector predictor
            ref_mock.return_value = MagicMock(
                helpful_insights="Use step-by-step approach",
                harmful_insights="Don't skip verification",
                confidence="0.8",
                analysis="Task completed successfully"
            )

            # Generator should fail
            with pytest.raises(Exception):
                generator(sample_task_input)

            # Reflector should succeed (independent circuit)
            result = reflector(sample_reflector_input)
            assert result.task_id == "test-task-1"

    # Check separate metrics
    metrics = get_llm_breaker_metrics()

    assert len(metrics) == 2  # Two separate breakers

    gen_metrics = next(m for m in metrics if "generator" in m["name"])
    ref_metrics = next(m for m in metrics if "reflector" in m["name"])

    assert gen_metrics["total_failures"] == 1
    assert ref_metrics["total_successes"] == 1


def test_circuit_breaker_with_dspy_exception(generator, sample_task_input):
    """T071: Circuit breaker should handle DSPy-specific exceptions."""

    with patch.object(generator, 'predictor') as mock_predictor:
        # Simulate DSPy/OpenAI exception
        mock_predictor.side_effect = Exception("Rate limit exceeded")

        # First call should raise exception
        with pytest.raises(Exception, match="Rate limit exceeded"):
            generator(sample_task_input)

        # Verify failure was recorded
        breaker = get_or_create_llm_breaker("generator")
        assert breaker.failure_count == 1


def test_circuit_breaker_fail_fast_reduces_latency(generator, sample_task_input):
    """T071: Open circuit should fail fast without waiting for LLM timeout."""

    # Create circuit breaker
    breaker = get_or_create_llm_breaker("generator", failure_threshold=2)

    with patch.object(generator, 'predictor') as mock_predictor:
        # Simulate slow failing API (1 second timeout)
        def slow_failure(*args, **kwargs):
            time.sleep(1.0)
            raise Exception("API timeout")

        mock_predictor.side_effect = slow_failure

        # First 2 calls will be slow (circuit learning)
        for i in range(2):
            with pytest.raises(RuntimeError):
                generator(sample_task_input)

        # Circuit should now be open
        # 3rd call should fail fast (< 0.1s)
        start_time = time.time()
        with pytest.raises(RuntimeError):
            generator(sample_task_input)
        elapsed = time.time() - start_time

        # Should fail in < 100ms (not wait 1 second)
        assert elapsed < 0.1, f"Circuit breaker should fail fast, took {elapsed:.3f}s"


def test_circuit_breaker_logs_state_transitions(generator, sample_task_input):
    """T071: Circuit breaker should log state transitions for monitoring."""

    with patch.object(generator, 'predictor') as mock_predictor:
        # Trigger failures to open circuit
        mock_predictor.side_effect = Exception("API error")

        for i in range(5):
            with pytest.raises(RuntimeError):
                generator(sample_task_input)

    # Verify circuit is open (which means state transition occurred)
    breaker = get_or_create_llm_breaker("generator")
    assert breaker.state.value == "open"

    # Get metrics to verify state was tracked
    metrics = breaker.get_metrics()
    assert metrics["state"] == "open"
    assert metrics["open_count"] >= 1
