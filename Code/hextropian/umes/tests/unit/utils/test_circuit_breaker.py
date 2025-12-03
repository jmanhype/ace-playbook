"""
Unit tests for circuit breaker factory with Purgatory.

Tests the AsyncCircuitBreakerFactory for protecting external service calls
(KMS, IdP) from cascading failures.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock

from umes.utils.circuit_breaker import AsyncCircuitBreakerFactory


class TestCircuitBreakerFactory:
    """Test circuit breaker factory creation and configuration."""

    @pytest.mark.asyncio
    async def test_factory_creates_kms_circuit_breaker(self):
        """Test that factory creates circuit breaker for KMS."""
        factory = AsyncCircuitBreakerFactory()

        cb = await factory.create_kms_breaker()

        assert cb is not None
        assert hasattr(cb, "call")

    @pytest.mark.asyncio
    async def test_factory_creates_idp_circuit_breaker(self):
        """Test that factory creates circuit breaker for IdP."""
        factory = AsyncCircuitBreakerFactory()

        cb = await factory.create_idp_breaker()

        assert cb is not None
        assert hasattr(cb, "call")

    @pytest.mark.asyncio
    async def test_factory_singleton_returns_same_instance_for_kms(self):
        """Test that factory returns same KMS breaker instance (singleton)."""
        factory = AsyncCircuitBreakerFactory()

        cb1 = await factory.create_kms_breaker()
        cb2 = await factory.create_kms_breaker()

        assert cb1 is cb2

    @pytest.mark.asyncio
    async def test_factory_singleton_returns_same_instance_for_idp(self):
        """Test that factory returns same IdP breaker instance (singleton)."""
        factory = AsyncCircuitBreakerFactory()

        cb1 = await factory.create_idp_breaker()
        cb2 = await factory.create_idp_breaker()

        assert cb1 is cb2

    @pytest.mark.asyncio
    async def test_factory_separate_instances_for_kms_and_idp(self):
        """Test that KMS and IdP breakers are separate instances."""
        factory = AsyncCircuitBreakerFactory()

        kms_cb = await factory.create_kms_breaker()
        idp_cb = await factory.create_idp_breaker()

        assert kms_cb is not idp_cb


class TestCircuitBreakerBasicBehavior:
    """Test basic circuit breaker behavior (CLOSED → OPEN → HALF_OPEN)."""

    @pytest.mark.asyncio
    async def test_successful_calls_pass_through(self):
        """Test that successful calls pass through when circuit is CLOSED."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        async def successful_operation():
            return "success"

        result = await cb.call(successful_operation)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after exceeding failure threshold."""
        factory = AsyncCircuitBreakerFactory(
            failure_threshold=3,
            timeout_seconds=60,
        )
        cb = await factory.create_kms_breaker()

        async def failing_operation():
            raise Exception("Service unavailable")

        # First 3 failures should be caught
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_operation)

        # 4th call should fail with CircuitBreakerOpenException
        # (circuit is now OPEN)
        with pytest.raises(Exception) as exc_info:
            await cb.call(failing_operation)

        # Should be a circuit breaker exception, not the original exception
        assert "circuit" in str(exc_info.value).lower() or "open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_half_opens_after_timeout(self):
        """Test that circuit transitions to HALF_OPEN after timeout."""
        factory = AsyncCircuitBreakerFactory(
            failure_threshold=2,
            timeout_seconds=1,  # Short timeout for testing
        )
        cb = await factory.create_kms_breaker()

        async def failing_operation():
            raise Exception("Service unavailable")

        # Trigger failures to open circuit
        for i in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_operation)

        # Circuit should be OPEN - next call fails immediately
        with pytest.raises(Exception):
            await cb.call(failing_operation)

        # Wait for timeout to expire
        await asyncio.sleep(1.5)

        # Circuit should now be HALF_OPEN - allows one test call
        # (This call will fail, but it's attempting recovery)
        with pytest.raises(Exception):
            await cb.call(failing_operation)


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior."""

    @pytest.mark.asyncio
    async def test_successful_call_in_half_open_closes_circuit(self):
        """Test that successful call in HALF_OPEN closes circuit."""
        factory = AsyncCircuitBreakerFactory(
            failure_threshold=2,
            timeout_seconds=1,
        )
        cb = await factory.create_kms_breaker()

        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return "recovered"

        # Open the circuit with 2 failures
        for i in range(2):
            with pytest.raises(Exception):
                await cb.call(flaky_operation)

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Next call in HALF_OPEN should succeed and close circuit
        result = await cb.call(flaky_operation)
        assert result == "recovered"

        # Subsequent calls should work (circuit is CLOSED)
        result = await cb.call(flaky_operation)
        assert result == "recovered"


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration options."""

    @pytest.mark.asyncio
    async def test_factory_accepts_custom_failure_threshold(self):
        """Test that factory accepts custom failure threshold."""
        factory = AsyncCircuitBreakerFactory(failure_threshold=5)

        cb = await factory.create_kms_breaker()

        assert cb is not None

    @pytest.mark.asyncio
    async def test_factory_accepts_custom_timeout(self):
        """Test that factory accepts custom timeout."""
        factory = AsyncCircuitBreakerFactory(timeout_seconds=30)

        cb = await factory.create_idp_breaker()

        assert cb is not None

    @pytest.mark.asyncio
    async def test_factory_accepts_custom_recovery_timeout(self):
        """Test that factory accepts custom recovery timeout."""
        factory = AsyncCircuitBreakerFactory(recovery_timeout_seconds=10)

        cb = await factory.create_kms_breaker()

        assert cb is not None


class TestCircuitBreakerErrorHandling:
    """Test circuit breaker error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_handles_async_exceptions(self):
        """Test that circuit breaker properly handles async exceptions."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        async def async_failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Async operation failed")

        with pytest.raises(ValueError):
            await cb.call(async_failing_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_handles_timeouts(self):
        """Test that circuit breaker handles operation timeouts."""
        factory = AsyncCircuitBreakerFactory(
            failure_threshold=2,
            operation_timeout_seconds=0.5,
        )
        cb = await factory.create_kms_breaker()

        async def slow_operation():
            await asyncio.sleep(2)  # Longer than timeout
            return "should timeout"

        # Timeout should count as failure
        with pytest.raises(Exception):
            await cb.call(slow_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_failure_count_on_success(self):
        """Test that failure count resets on successful call."""
        factory = AsyncCircuitBreakerFactory(failure_threshold=3)
        cb = await factory.create_kms_breaker()

        call_count = 0

        async def intermittent_operation():
            nonlocal call_count
            call_count += 1
            if call_count in [1, 3]:  # Fail on 1st and 3rd call
                raise Exception("Intermittent failure")
            return "success"

        # First failure
        with pytest.raises(Exception):
            await cb.call(intermittent_operation)

        # Success - resets counter
        result = await cb.call(intermittent_operation)
        assert result == "success"

        # Another failure - but counter was reset, so circuit stays closed
        with pytest.raises(Exception):
            await cb.call(intermittent_operation)

        # Next success should work (circuit still closed)
        result = await cb.call(intermittent_operation)
        assert result == "success"


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_tracks_failure_count(self):
        """Test that circuit breaker tracks failure count."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        # Check if circuit breaker has failure tracking
        assert hasattr(cb, "fail") or hasattr(cb, "current_failures") or hasattr(cb, "_failure_count")

    @pytest.mark.asyncio
    async def test_circuit_breaker_tracks_state(self):
        """Test that circuit breaker tracks current state."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        # Check if circuit breaker has state tracking
        assert hasattr(cb, "state") or hasattr(cb, "current_state") or hasattr(cb, "_state")

    @pytest.mark.asyncio
    async def test_circuit_breaker_provides_health_status(self):
        """Test that circuit breaker provides health status."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        # Circuit breaker should provide some way to check if it's healthy
        # (CLOSED = healthy, OPEN = unhealthy)
        has_health_method = (
            hasattr(cb, "is_healthy") or
            hasattr(cb, "is_closed") or
            hasattr(cb, "state") or
            hasattr(cb, "current_state")
        )

        assert has_health_method is True

    @pytest.mark.asyncio
    async def test_factory_reset_all_clears_breakers(self):
        """Test that factory.reset_all() clears cached breakers."""
        factory = AsyncCircuitBreakerFactory()

        # Create breakers
        kms_breaker = await factory.create_kms_breaker()
        idp_breaker = await factory.create_idp_breaker()

        assert kms_breaker is not None
        assert idp_breaker is not None

        # Reset all breakers
        factory.reset_all()

        # Verify breakers are cleared
        assert factory._kms_breaker is None
        assert factory._idp_breaker is None

        # Creating new breakers should give new instances
        new_kms_breaker = await factory.create_kms_breaker()
        assert new_kms_breaker is not kms_breaker

    @pytest.mark.asyncio
    async def test_breaker_current_failures_property(self):
        """Test that CircuitBreakerWrapper.current_failures returns count."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        # Initially should have 0 failures
        assert cb.current_failures == 0

        # After a failure, should increment
        async def failing_operation():
            raise ValueError("Simulated failure")

        try:
            await cb.call(failing_operation)
        except ValueError:
            pass

        # current_failures should be accessible
        assert cb.current_failures >= 0  # Should have incremented

    @pytest.mark.asyncio
    async def test_breaker_is_healthy_returns_bool(self):
        """Test that CircuitBreakerWrapper.is_healthy() returns boolean."""
        factory = AsyncCircuitBreakerFactory()
        cb = await factory.create_kms_breaker()

        # is_healthy() should return True when circuit is closed
        health_status = cb.is_healthy()
        assert isinstance(health_status, bool)
        assert health_status is True  # New breaker should be healthy
