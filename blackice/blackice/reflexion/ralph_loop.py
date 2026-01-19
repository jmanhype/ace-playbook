"""Ralph Loop for BLACKICE 3.0.

Implements the try-fail-reflect-learn-retry pattern for iterative improvement.
Ralph = Reflective Autonomous Learning & Progress Handler
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from blackice.core.retry import RetryConfig
from blackice.instrumentation import get_logger
from blackice.primitives.errors import BlackiceError

logger = get_logger(__name__)

T = TypeVar("T")


class LoopPhase(str, Enum):
    """Phases of the Ralph loop."""

    TRY = "try"
    FAIL = "fail"
    REFLECT = "reflect"
    LEARN = "learn"
    RETRY = "retry"
    SUCCESS = "success"
    EXHAUSTED = "exhausted"


@dataclass
class Attempt:
    """Record of a single attempt in the loop."""

    attempt_number: int
    phase: LoopPhase
    started_at: float
    completed_at: float | None = None
    duration_seconds: float | None = None
    error: str | None = None
    reflection: str | None = None
    learnings: list[str] = field(default_factory=list)
    adjustments: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class LoopResult(Generic[T]):
    """Result of running the Ralph loop."""

    success: bool
    result: T | None
    attempts: list[Attempt]
    total_duration: float
    final_phase: LoopPhase

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def learnings(self) -> list[str]:
        """All learnings accumulated across attempts."""
        all_learnings = []
        for attempt in self.attempts:
            all_learnings.extend(attempt.learnings)
        return all_learnings


@dataclass
class LoopConfig:
    """Configuration for the Ralph loop."""

    max_attempts: int = 5
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Callbacks for each phase
    on_try: Callable[[int], None] | None = None
    on_fail: Callable[[int, Exception], None] | None = None
    on_reflect: Callable[[int, Exception, str | None], None] | None = None
    on_learn: Callable[[int, list[str]], None] | None = None
    on_retry: Callable[[int, dict[str, Any]], None] | None = None


class ReflectionProvider:
    """Provider for generating reflections on failures.

    Can be subclassed to use LLM-based reflection.
    """

    async def reflect(
        self,
        error: Exception,
        context: dict[str, Any],
        attempt: int,
    ) -> str:
        """Generate a reflection on why the attempt failed.

        Args:
            error: The exception that was raised
            context: Additional context about the attempt
            attempt: The attempt number

        Returns:
            A reflection string analyzing the failure
        """
        return f"Attempt {attempt} failed with {type(error).__name__}: {error}"

    async def extract_learnings(
        self,
        reflection: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Extract actionable learnings from a reflection.

        Args:
            reflection: The reflection text
            context: Additional context

        Returns:
            List of actionable learnings
        """
        # Default implementation returns generic learnings
        return [f"Failure observed: {reflection[:100]}"]

    async def suggest_adjustments(
        self,
        learnings: list[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Suggest adjustments based on learnings.

        Args:
            learnings: The learnings extracted
            context: Additional context

        Returns:
            Dictionary of suggested parameter adjustments
        """
        return {}


class RalphLoop(Generic[T]):
    """The Ralph loop: try-fail-reflect-learn-retry.

    Implements iterative improvement through:
    1. TRY: Execute the operation
    2. FAIL: Capture the failure
    3. REFLECT: Analyze what went wrong
    4. LEARN: Extract actionable learnings
    5. RETRY: Apply learnings and try again

    Example:
        ```python
        async def risky_operation(params: dict) -> str:
            # ... operation that might fail
            return result

        loop = RalphLoop(
            operation=risky_operation,
            config=LoopConfig(max_attempts=5),
        )
        result = await loop.run(initial_params={"key": "value"})
        if result.success:
            print(f"Succeeded after {result.attempt_count} attempts")
            print(f"Learnings: {result.learnings}")
        ```
    """

    def __init__(
        self,
        operation: Callable[..., T] | Callable[..., Any],
        config: LoopConfig | None = None,
        reflection_provider: ReflectionProvider | None = None,
    ) -> None:
        """Initialize the Ralph loop.

        Args:
            operation: The async operation to execute
            config: Loop configuration
            reflection_provider: Provider for generating reflections
        """
        self.operation = operation
        self.config = config or LoopConfig()
        self.reflection_provider = reflection_provider or ReflectionProvider()
        self._attempts: list[Attempt] = []
        self._current_phase = LoopPhase.TRY
        self._context: dict[str, Any] = {}

    @property
    def attempts(self) -> list[Attempt]:
        """Get all attempts made so far."""
        return self._attempts.copy()

    @property
    def current_phase(self) -> LoopPhase:
        """Get the current phase."""
        return self._current_phase

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before retry using exponential backoff."""
        import random

        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay,
        )
        if self.config.jitter:
            delay *= 0.5 + random.random()
        return delay

    async def run(
        self,
        initial_params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> LoopResult[T]:
        """Run the Ralph loop.

        Args:
            initial_params: Initial parameters for the operation
            context: Additional context for reflection

        Returns:
            LoopResult with success status and all attempts
        """
        self._attempts = []
        self._context = context or {}
        params = initial_params or {}
        loop_start = time.monotonic()

        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt = Attempt(
                attempt_number=attempt_num,
                phase=LoopPhase.TRY,
                started_at=time.time(),
            )

            try:
                # TRY phase
                self._current_phase = LoopPhase.TRY
                if self.config.on_try:
                    self.config.on_try(attempt_num)

                logger.info(
                    "Ralph loop attempting operation",
                    attempt=attempt_num,
                    max_attempts=self.config.max_attempts,
                    params_keys=list(params.keys()),
                )

                # Execute operation
                if asyncio.iscoroutinefunction(self.operation):
                    result = await self.operation(**params)
                else:
                    result = self.operation(**params)

                # SUCCESS!
                attempt.phase = LoopPhase.SUCCESS
                attempt.completed_at = time.time()
                attempt.duration_seconds = attempt.completed_at - attempt.started_at
                attempt.result = result
                self._attempts.append(attempt)
                self._current_phase = LoopPhase.SUCCESS

                logger.info(
                    "Ralph loop succeeded",
                    attempt=attempt_num,
                    duration=attempt.duration_seconds,
                )

                return LoopResult(
                    success=True,
                    result=result,
                    attempts=self._attempts,
                    total_duration=time.monotonic() - loop_start,
                    final_phase=LoopPhase.SUCCESS,
                )

            except Exception as e:
                # FAIL phase
                self._current_phase = LoopPhase.FAIL
                attempt.phase = LoopPhase.FAIL
                attempt.error = str(e)
                if self.config.on_fail:
                    self.config.on_fail(attempt_num, e)

                logger.warning(
                    "Ralph loop attempt failed",
                    attempt=attempt_num,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                # Check if we've exhausted attempts
                if attempt_num >= self.config.max_attempts:
                    attempt.completed_at = time.time()
                    attempt.duration_seconds = attempt.completed_at - attempt.started_at
                    self._attempts.append(attempt)
                    self._current_phase = LoopPhase.EXHAUSTED

                    logger.error(
                        "Ralph loop exhausted all attempts",
                        total_attempts=attempt_num,
                        final_error=str(e),
                    )

                    return LoopResult(
                        success=False,
                        result=None,
                        attempts=self._attempts,
                        total_duration=time.monotonic() - loop_start,
                        final_phase=LoopPhase.EXHAUSTED,
                    )

                # REFLECT phase
                self._current_phase = LoopPhase.REFLECT
                attempt.phase = LoopPhase.REFLECT
                reflection = await self.reflection_provider.reflect(
                    error=e,
                    context={**self._context, "params": params, "attempt": attempt_num},
                    attempt=attempt_num,
                )
                attempt.reflection = reflection
                if self.config.on_reflect:
                    self.config.on_reflect(attempt_num, e, reflection)

                logger.info(
                    "Ralph loop reflected on failure",
                    attempt=attempt_num,
                    reflection_length=len(reflection),
                )

                # LEARN phase
                self._current_phase = LoopPhase.LEARN
                attempt.phase = LoopPhase.LEARN
                learnings = await self.reflection_provider.extract_learnings(
                    reflection=reflection,
                    context={**self._context, "params": params},
                )
                attempt.learnings = learnings
                if self.config.on_learn:
                    self.config.on_learn(attempt_num, learnings)

                logger.info(
                    "Ralph loop extracted learnings",
                    attempt=attempt_num,
                    learning_count=len(learnings),
                )

                # RETRY phase - suggest adjustments
                self._current_phase = LoopPhase.RETRY
                attempt.phase = LoopPhase.RETRY
                adjustments = await self.reflection_provider.suggest_adjustments(
                    learnings=learnings,
                    context={**self._context, "params": params},
                )
                attempt.adjustments = adjustments
                if self.config.on_retry:
                    self.config.on_retry(attempt_num, adjustments)

                # Apply adjustments to params
                params = {**params, **adjustments}

                attempt.completed_at = time.time()
                attempt.duration_seconds = attempt.completed_at - attempt.started_at
                self._attempts.append(attempt)

                # Wait before retry
                delay = self._calculate_delay(attempt_num)
                logger.info(
                    "Ralph loop waiting before retry",
                    attempt=attempt_num,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but handle gracefully
        self._current_phase = LoopPhase.EXHAUSTED
        return LoopResult(
            success=False,
            result=None,
            attempts=self._attempts,
            total_duration=time.monotonic() - loop_start,
            final_phase=LoopPhase.EXHAUSTED,
        )


class LLMReflectionProvider(ReflectionProvider):
    """Reflection provider that uses an LLM for analysis.

    Integrates with model providers to generate intelligent
    reflections and learnings from failures.
    """

    def __init__(
        self,
        model_provider: Any,  # ModelProvider
        reflection_prompt_template: str | None = None,
        learning_prompt_template: str | None = None,
    ) -> None:
        """Initialize with a model provider.

        Args:
            model_provider: The model provider to use for reflection
            reflection_prompt_template: Custom prompt for reflection
            learning_prompt_template: Custom prompt for learning extraction
        """
        self.model_provider = model_provider
        self.reflection_prompt_template = reflection_prompt_template or (
            "Analyze this failure and explain what went wrong:\n\n"
            "Error: {error}\n"
            "Error Type: {error_type}\n"
            "Attempt: {attempt}\n"
            "Context: {context}\n\n"
            "Provide a concise analysis of the root cause."
        )
        self.learning_prompt_template = learning_prompt_template or (
            "Based on this reflection, extract actionable learnings:\n\n"
            "{reflection}\n\n"
            "List 1-3 specific, actionable changes that could prevent this failure."
        )

    async def reflect(
        self,
        error: Exception,
        context: dict[str, Any],
        attempt: int,
    ) -> str:
        """Generate LLM-powered reflection on failure."""
        prompt = self.reflection_prompt_template.format(
            error=str(error),
            error_type=type(error).__name__,
            attempt=attempt,
            context=str(context)[:500],  # Limit context size
        )

        try:
            result = await self.model_provider.generate(prompt)
            return result.content
        except Exception as e:
            # Fallback to basic reflection if LLM fails
            logger.warning("LLM reflection failed, using fallback", error=str(e))
            return await super().reflect(error, context, attempt)

    async def extract_learnings(
        self,
        reflection: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Extract learnings using LLM."""
        prompt = self.learning_prompt_template.format(
            reflection=reflection,
        )

        try:
            result = await self.model_provider.generate(prompt)
            # Parse learnings from response (simple line-based parsing)
            lines = result.content.strip().split("\n")
            learnings = [
                line.strip().lstrip("- ").lstrip("• ").lstrip("* ")
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
            return learnings[:5]  # Limit to 5 learnings
        except Exception as e:
            logger.warning("LLM learning extraction failed", error=str(e))
            return await super().extract_learnings(reflection, context)


async def with_ralph_loop(
    operation: Callable[..., T],
    *,
    max_attempts: int = 5,
    reflection_provider: ReflectionProvider | None = None,
    **params: Any,
) -> LoopResult[T]:
    """Convenience function to run an operation with Ralph loop.

    Args:
        operation: The operation to run
        max_attempts: Maximum number of attempts
        reflection_provider: Optional reflection provider
        **params: Parameters to pass to the operation

    Returns:
        LoopResult with success status
    """
    loop = RalphLoop(
        operation=operation,
        config=LoopConfig(max_attempts=max_attempts),
        reflection_provider=reflection_provider,
    )
    return await loop.run(initial_params=params)
