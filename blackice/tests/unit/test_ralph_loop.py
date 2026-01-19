"""Unit tests for the Ralph Loop (try-fail-reflect-learn-retry pattern).

Tests the RalphLoop class, ReflectionProvider, and related components
per FR-001 (Ralph pattern execution).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blackice.reflexion.ralph_loop import (
    Attempt,
    LLMReflectionProvider,
    LoopConfig,
    LoopPhase,
    LoopResult,
    RalphLoop,
    ReflectionProvider,
    with_ralph_loop,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_fail_n_times_operation(fail_count: int = 2, success_value: str = "success"):
    """Create an operation that fails N times then succeeds."""
    state = {"call_count": 0}

    async def operation(**kwargs: Any) -> str:
        state["call_count"] += 1
        if state["call_count"] <= fail_count:
            raise ValueError(f"Intentional failure {state['call_count']}")
        return success_value

    # Attach state for inspection
    operation.state = state  # type: ignore
    return operation


def make_always_fail_operation():
    """Create an operation that always fails."""
    state = {"call_count": 0}

    async def operation(**kwargs: Any) -> str:
        state["call_count"] += 1
        raise RuntimeError(f"Always fails: attempt {state['call_count']}")

    # Attach state for inspection
    operation.state = state  # type: ignore
    return operation


@pytest.fixture
def loop_config() -> LoopConfig:
    """Create a test loop config with fast delays."""
    return LoopConfig(
        max_attempts=3,
        initial_delay=0.01,  # Fast for testing
        max_delay=0.05,
        jitter=False,  # Disable jitter for predictable tests
    )


# =============================================================================
# RalphLoop Core Tests
# =============================================================================


class TestRalphLoop:
    """Tests for the RalphLoop class."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, loop_config: LoopConfig) -> None:
        """Loop succeeds immediately when operation succeeds."""
        async def successful_op(**kwargs: Any) -> str:
            return "immediate_success"

        loop = RalphLoop(operation=successful_op, config=loop_config)
        result = await loop.run()

        assert result.success is True
        assert result.result == "immediate_success"
        assert result.attempt_count == 1
        assert result.final_phase == LoopPhase.SUCCESS
        assert len(result.attempts) == 1
        assert result.attempts[0].phase == LoopPhase.SUCCESS

    @pytest.mark.asyncio
    async def test_success_after_retries(self, loop_config: LoopConfig) -> None:
        """Loop succeeds after failing and retrying."""
        operation = make_fail_n_times_operation(fail_count=2, success_value="eventual_success")
        loop = RalphLoop(operation=operation, config=loop_config)
        result = await loop.run()

        assert result.success is True
        assert result.result == "eventual_success"
        assert result.attempt_count == 3
        assert result.final_phase == LoopPhase.SUCCESS
        assert operation.state["call_count"] == 3

    @pytest.mark.asyncio
    async def test_exhausted_after_max_attempts(self, loop_config: LoopConfig) -> None:
        """Loop returns exhausted when all attempts fail."""
        operation = make_always_fail_operation()
        loop = RalphLoop(operation=operation, config=loop_config)
        result = await loop.run()

        assert result.success is False
        assert result.result is None
        assert result.attempt_count == 3
        assert result.final_phase == LoopPhase.EXHAUSTED
        assert operation.state["call_count"] == 3

    @pytest.mark.asyncio
    async def test_initial_params_passed_to_operation(
        self, loop_config: LoopConfig
    ) -> None:
        """Initial params are passed to the operation."""
        received_params: dict[str, Any] = {}

        async def capture_params(**kwargs: Any) -> str:
            received_params.update(kwargs)
            return "ok"

        loop = RalphLoop(operation=capture_params, config=loop_config)
        await loop.run(initial_params={"key1": "value1", "key2": 42})

        assert received_params == {"key1": "value1", "key2": 42}

    @pytest.mark.asyncio
    async def test_sync_operation_supported(self, loop_config: LoopConfig) -> None:
        """Loop supports synchronous operations."""
        def sync_op(**kwargs: Any) -> str:
            return "sync_result"

        loop = RalphLoop(operation=sync_op, config=loop_config)
        result = await loop.run()

        assert result.success is True
        assert result.result == "sync_result"

    @pytest.mark.asyncio
    async def test_attempt_records_error_info(self, loop_config: LoopConfig) -> None:
        """Failed attempts record error information."""
        operation = make_fail_n_times_operation(fail_count=1)
        loop = RalphLoop(operation=operation, config=loop_config)
        result = await loop.run()

        # First attempt should have error info
        first_attempt = result.attempts[0]
        assert first_attempt.error is not None
        assert "Intentional failure" in first_attempt.error
        assert first_attempt.reflection is not None

    @pytest.mark.asyncio
    async def test_learnings_accumulated(self, loop_config: LoopConfig) -> None:
        """Learnings are accumulated across attempts."""
        operation = make_fail_n_times_operation(fail_count=2)
        loop = RalphLoop(operation=operation, config=loop_config)
        result = await loop.run()

        # Should have learnings from the two failed attempts
        assert len(result.learnings) >= 2

    @pytest.mark.asyncio
    async def test_duration_tracked(self, loop_config: LoopConfig) -> None:
        """Duration is tracked for loop and individual attempts."""
        async def slow_op(**kwargs: Any) -> str:
            await asyncio.sleep(0.01)
            return "done"

        loop = RalphLoop(operation=slow_op, config=loop_config)
        result = await loop.run()

        assert result.total_duration > 0
        assert result.attempts[0].duration_seconds is not None
        assert result.attempts[0].duration_seconds > 0

    @pytest.mark.asyncio
    async def test_current_phase_property(self, loop_config: LoopConfig) -> None:
        """Current phase property reflects loop state."""
        operation = make_fail_n_times_operation(fail_count=1)
        loop = RalphLoop(operation=operation, config=loop_config)

        # Before run
        assert loop.current_phase == LoopPhase.TRY

        # After run
        await loop.run()
        assert loop.current_phase == LoopPhase.SUCCESS


# =============================================================================
# LoopConfig Tests
# =============================================================================


class TestLoopConfig:
    """Tests for LoopConfig callbacks."""

    @pytest.mark.asyncio
    async def test_on_try_callback(self) -> None:
        """on_try callback is called for each attempt."""
        try_calls: list[int] = []

        config = LoopConfig(
            max_attempts=3,
            initial_delay=0.01,
            on_try=lambda n: try_calls.append(n),
        )

        async def always_success(**kwargs: Any) -> str:
            return "ok"

        loop = RalphLoop(operation=always_success, config=config)
        await loop.run()

        assert try_calls == [1]  # Only one attempt needed

    @pytest.mark.asyncio
    async def test_on_fail_callback(self) -> None:
        """on_fail callback is called on failures."""
        fail_calls: list[tuple[int, Exception]] = []

        config = LoopConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False,
            on_fail=lambda n, e: fail_calls.append((n, e)),
        )

        operation = make_always_fail_operation()
        loop = RalphLoop(operation=operation, config=config)
        await loop.run()

        assert len(fail_calls) == 2
        assert fail_calls[0][0] == 1
        assert fail_calls[1][0] == 2

    @pytest.mark.asyncio
    async def test_on_reflect_callback(self) -> None:
        """on_reflect callback is called after reflection."""
        reflect_calls: list[tuple[int, str | None]] = []

        config = LoopConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False,
            on_reflect=lambda n, e, r: reflect_calls.append((n, r)),
        )

        operation = make_fail_n_times_operation(fail_count=1)
        loop = RalphLoop(operation=operation, config=config)
        await loop.run()

        # Only called for non-final failures (retry happens)
        assert len(reflect_calls) == 1
        assert reflect_calls[0][0] == 1
        assert reflect_calls[0][1] is not None

    @pytest.mark.asyncio
    async def test_on_learn_callback(self) -> None:
        """on_learn callback is called with learnings."""
        learn_calls: list[tuple[int, list[str]]] = []

        config = LoopConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False,
            on_learn=lambda n, l: learn_calls.append((n, l)),
        )

        operation = make_fail_n_times_operation(fail_count=1)
        loop = RalphLoop(operation=operation, config=config)
        await loop.run()

        assert len(learn_calls) == 1
        assert learn_calls[0][0] == 1
        assert len(learn_calls[0][1]) > 0

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        """on_retry callback is called with adjustments."""
        retry_calls: list[tuple[int, dict[str, Any]]] = []

        config = LoopConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False,
            on_retry=lambda n, a: retry_calls.append((n, a)),
        )

        operation = make_fail_n_times_operation(fail_count=1)
        loop = RalphLoop(operation=operation, config=config)
        await loop.run()

        assert len(retry_calls) == 1
        assert retry_calls[0][0] == 1


# =============================================================================
# ReflectionProvider Tests
# =============================================================================


class TestReflectionProvider:
    """Tests for the default ReflectionProvider."""

    @pytest.mark.asyncio
    async def test_reflect_returns_error_summary(self) -> None:
        """Default reflection returns error summary."""
        provider = ReflectionProvider()
        error = ValueError("test error message")

        reflection = await provider.reflect(
            error=error,
            context={"key": "value"},
            attempt=1,
        )

        assert "ValueError" in reflection
        assert "test error message" in reflection
        assert "1" in reflection  # attempt number

    @pytest.mark.asyncio
    async def test_extract_learnings_returns_list(self) -> None:
        """Default learning extraction returns a list."""
        provider = ReflectionProvider()

        learnings = await provider.extract_learnings(
            reflection="Something failed because of X",
            context={},
        )

        assert isinstance(learnings, list)
        assert len(learnings) > 0

    @pytest.mark.asyncio
    async def test_suggest_adjustments_returns_empty(self) -> None:
        """Default adjustment suggestion returns empty dict."""
        provider = ReflectionProvider()

        adjustments = await provider.suggest_adjustments(
            learnings=["learning 1"],
            context={},
        )

        assert adjustments == {}


# =============================================================================
# LLMReflectionProvider Tests
# =============================================================================


class TestLLMReflectionProvider:
    """Tests for the LLM-based ReflectionProvider."""

    @pytest.mark.asyncio
    async def test_reflect_uses_model_provider(self) -> None:
        """LLM provider calls model for reflection."""
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = MagicMock(content="LLM reflection")

        llm_reflection = LLMReflectionProvider(model_provider=mock_provider)
        result = await llm_reflection.reflect(
            error=ValueError("test"),
            context={},
            attempt=1,
        )

        assert result == "LLM reflection"
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_falls_back_on_error(self) -> None:
        """LLM provider falls back to default on model error."""
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("Model unavailable")

        llm_reflection = LLMReflectionProvider(model_provider=mock_provider)
        result = await llm_reflection.reflect(
            error=ValueError("test error"),
            context={},
            attempt=1,
        )

        # Should fall back to basic reflection
        assert "ValueError" in result
        assert "test error" in result

    @pytest.mark.asyncio
    async def test_extract_learnings_parses_response(self) -> None:
        """LLM provider parses learnings from model response."""
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = MagicMock(
            content="- Learning one\n- Learning two\n* Learning three"
        )

        llm_reflection = LLMReflectionProvider(model_provider=mock_provider)
        learnings = await llm_reflection.extract_learnings(
            reflection="test reflection",
            context={},
        )

        assert len(learnings) == 3
        assert "Learning one" in learnings[0]
        assert "Learning two" in learnings[1]
        assert "Learning three" in learnings[2]

    @pytest.mark.asyncio
    async def test_custom_prompt_templates(self) -> None:
        """Custom prompt templates are used."""
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = MagicMock(content="custom response")

        llm_reflection = LLMReflectionProvider(
            model_provider=mock_provider,
            reflection_prompt_template="Custom: {error}",
        )
        await llm_reflection.reflect(
            error=ValueError("test"),
            context={},
            attempt=1,
        )

        call_args = mock_provider.generate.call_args[0][0]
        assert "Custom:" in call_args


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestWithRalphLoop:
    """Tests for the with_ralph_loop convenience function."""

    @pytest.mark.asyncio
    async def test_runs_operation_with_loop(self) -> None:
        """Convenience function runs operation with Ralph loop."""
        call_count = 0

        async def simple_op(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        result = await with_ralph_loop(simple_op, max_attempts=3)

        assert result.success is True
        assert result.result == "result_1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_operation(self) -> None:
        """Kwargs are passed to the operation."""
        received: dict[str, Any] = {}

        async def capture_op(**kwargs: Any) -> str:
            received.update(kwargs)
            return "ok"

        await with_ralph_loop(capture_op, max_attempts=2, param1="value1", param2=99)

        assert received == {"param1": "value1", "param2": 99}

    @pytest.mark.asyncio
    async def test_uses_custom_reflection_provider(self) -> None:
        """Custom reflection provider is used."""
        custom_provider = ReflectionProvider()

        async def failing_op(**kwargs: Any) -> str:
            raise ValueError("fail")

        result = await with_ralph_loop(
            failing_op,
            max_attempts=2,
            reflection_provider=custom_provider,
        )

        assert result.success is False
        assert result.attempt_count == 2


# =============================================================================
# LoopResult Tests
# =============================================================================


class TestLoopResult:
    """Tests for LoopResult properties."""

    def test_attempt_count_property(self) -> None:
        """attempt_count returns number of attempts."""
        result = LoopResult(
            success=True,
            result="test",
            attempts=[
                Attempt(attempt_number=1, phase=LoopPhase.SUCCESS, started_at=0),
                Attempt(attempt_number=2, phase=LoopPhase.SUCCESS, started_at=0),
            ],
            total_duration=1.0,
            final_phase=LoopPhase.SUCCESS,
        )

        assert result.attempt_count == 2

    def test_learnings_aggregated(self) -> None:
        """learnings property aggregates from all attempts."""
        result = LoopResult(
            success=True,
            result="test",
            attempts=[
                Attempt(
                    attempt_number=1,
                    phase=LoopPhase.RETRY,
                    started_at=0,
                    learnings=["learning1", "learning2"],
                ),
                Attempt(
                    attempt_number=2,
                    phase=LoopPhase.SUCCESS,
                    started_at=0,
                    learnings=["learning3"],
                ),
            ],
            total_duration=1.0,
            final_phase=LoopPhase.SUCCESS,
        )

        assert result.learnings == ["learning1", "learning2", "learning3"]


# =============================================================================
# Delay Calculation Tests
# =============================================================================


class TestDelayCalculation:
    """Tests for exponential backoff delay calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        """Delay increases exponentially with attempts."""
        config = LoopConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=False,
        )

        loop = RalphLoop(operation=lambda **k: None, config=config)

        delay1 = loop._calculate_delay(1)
        delay2 = loop._calculate_delay(2)
        delay3 = loop._calculate_delay(3)

        assert delay2 > delay1
        assert delay3 > delay2
        # With base 2: delay1=2, delay2=4, delay3=8
        assert delay2 == delay1 * 2
        assert delay3 == delay2 * 2

    @pytest.mark.asyncio
    async def test_max_delay_respected(self) -> None:
        """Delay does not exceed max_delay."""
        config = LoopConfig(
            initial_delay=1.0,
            exponential_base=10.0,
            max_delay=5.0,
            jitter=False,
        )

        loop = RalphLoop(operation=lambda **k: None, config=config)

        delay = loop._calculate_delay(100)  # Very high attempt number

        assert delay <= 5.0

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self) -> None:
        """Jitter adds randomness to delays."""
        config = LoopConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=True,
        )

        loop = RalphLoop(operation=lambda **k: None, config=config)

        # Run multiple times to check for variation
        delays = [loop._calculate_delay(1) for _ in range(10)]

        # With jitter, not all delays should be exactly the same
        unique_delays = set(delays)
        assert len(unique_delays) > 1
