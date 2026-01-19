"""Run State Machine for BLACKICE 3.0.

Implements the state machine for run lifecycle management:
pending -> planning -> executing -> verifying -> completed/failed
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from blackice.instrumentation import get_logger
from blackice.primitives.errors import StateError

logger = get_logger(__name__)


class RunState(str, Enum):
    """States in the run lifecycle."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid state transitions
TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.PENDING: {RunState.PLANNING, RunState.CANCELLED},
    RunState.PLANNING: {RunState.EXECUTING, RunState.FAILED, RunState.CANCELLED},
    RunState.EXECUTING: {
        RunState.VERIFYING,
        RunState.PAUSED,
        RunState.FAILED,
        RunState.CANCELLED,
    },
    RunState.VERIFYING: {RunState.COMPLETED, RunState.EXECUTING, RunState.FAILED},
    RunState.PAUSED: {RunState.RESUMING, RunState.CANCELLED},
    RunState.RESUMING: {RunState.EXECUTING, RunState.FAILED},
    RunState.COMPLETED: set(),  # Terminal state
    RunState.FAILED: {RunState.PENDING},  # Can retry from failed
    RunState.CANCELLED: set(),  # Terminal state
}


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: RunState
    to_state: RunState
    timestamp: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Context for a run."""

    run_id: str
    vision: str
    state: RunState = RunState.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    transitions: list[StateTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if run is in a terminal state."""
        return self.state in (RunState.COMPLETED, RunState.CANCELLED)

    @property
    def can_retry(self) -> bool:
        """Check if run can be retried."""
        return self.state == RunState.FAILED

    @property
    def duration_seconds(self) -> float:
        """Get run duration in seconds."""
        return self.updated_at - self.created_at


StateHandler = Callable[[RunContext], Any]


class RunStateMachine:
    """State machine for managing run lifecycle.

    Handles state transitions, validation, and event emission.

    Example:
        ```python
        machine = RunStateMachine()
        ctx = machine.create_run("run-001", "Build an API")

        # Transition through states
        await machine.transition(ctx, RunState.PLANNING)
        await machine.transition(ctx, RunState.EXECUTING)
        await machine.transition(ctx, RunState.VERIFYING)
        await machine.transition(ctx, RunState.COMPLETED)
        ```
    """

    def __init__(self) -> None:
        """Initialize the state machine."""
        self._runs: dict[str, RunContext] = {}
        self._handlers: dict[RunState, list[StateHandler]] = {
            state: [] for state in RunState
        }
        self._transition_handlers: list[Callable[[StateTransition], Any]] = []

    def create_run(
        self,
        run_id: str,
        vision: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunContext:
        """Create a new run in pending state.

        Args:
            run_id: Unique identifier for the run
            vision: The vision/description for this run
            metadata: Additional metadata

        Returns:
            RunContext for the new run
        """
        if run_id in self._runs:
            raise StateError(
                f"Run {run_id} already exists",
                context={"run_id": run_id},
            )

        ctx = RunContext(
            run_id=run_id,
            vision=vision,
            metadata=metadata or {},
        )

        self._runs[run_id] = ctx

        logger.info(
            "Run created",
            run_id=run_id,
            state=ctx.state.value,
        )

        return ctx

    def get_run(self, run_id: str) -> RunContext | None:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        state: RunState | None = None,
    ) -> list[RunContext]:
        """List all runs, optionally filtered by state."""
        runs = list(self._runs.values())
        if state is not None:
            runs = [r for r in runs if r.state == state]
        return runs

    async def transition(
        self,
        ctx: RunContext,
        to_state: RunState,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Transition a run to a new state.

        Args:
            ctx: The run context
            to_state: Target state
            reason: Reason for transition
            metadata: Additional metadata

        Returns:
            StateTransition record

        Raises:
            StateError: If transition is not valid
        """
        from_state = ctx.state

        # Validate transition
        if to_state not in TRANSITIONS.get(from_state, set()):
            raise StateError(
                f"Invalid transition from {from_state.value} to {to_state.value}",
                context={
                    "run_id": ctx.run_id,
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "valid_transitions": [s.value for s in TRANSITIONS.get(from_state, set())],
                },
            )

        # Create transition record
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=time.time(),
            reason=reason,
            metadata=metadata or {},
        )

        # Update context
        ctx.state = to_state
        ctx.updated_at = time.time()
        ctx.transitions.append(transition)

        logger.info(
            "State transition",
            run_id=ctx.run_id,
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
        )

        # Call state handlers
        for handler in self._handlers.get(to_state, []):
            try:
                result = handler(ctx)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "State handler failed",
                    state=to_state.value,
                    error=str(e),
                )

        # Call transition handlers
        for handler in self._transition_handlers:
            try:
                result = handler(transition)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Transition handler failed",
                    error=str(e),
                )

        return transition

    def on_state(self, state: RunState, handler: StateHandler) -> None:
        """Register a handler for when runs enter a state.

        Args:
            state: The state to handle
            handler: Handler function
        """
        self._handlers[state].append(handler)

    def on_transition(
        self,
        handler: Callable[[StateTransition], Any],
    ) -> None:
        """Register a handler for all transitions.

        Args:
            handler: Handler function
        """
        self._transition_handlers.append(handler)

    async def fail(
        self,
        ctx: RunContext,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Transition a run to failed state.

        Args:
            ctx: The run context
            error: Error message
            metadata: Additional metadata

        Returns:
            StateTransition record
        """
        ctx.error = error
        return await self.transition(
            ctx,
            RunState.FAILED,
            reason=error,
            metadata=metadata,
        )

    async def complete(
        self,
        ctx: RunContext,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Transition a run to completed state.

        Args:
            ctx: The run context
            metadata: Additional metadata

        Returns:
            StateTransition record
        """
        return await self.transition(
            ctx,
            RunState.COMPLETED,
            reason="Run completed successfully",
            metadata=metadata,
        )

    async def cancel(
        self,
        ctx: RunContext,
        reason: str = "Cancelled by user",
    ) -> StateTransition:
        """Cancel a run.

        Args:
            ctx: The run context
            reason: Cancellation reason

        Returns:
            StateTransition record
        """
        return await self.transition(
            ctx,
            RunState.CANCELLED,
            reason=reason,
        )

    async def pause(
        self,
        ctx: RunContext,
        reason: str = "Paused by user",
    ) -> StateTransition:
        """Pause a run.

        Args:
            ctx: The run context
            reason: Pause reason

        Returns:
            StateTransition record
        """
        return await self.transition(
            ctx,
            RunState.PAUSED,
            reason=reason,
        )

    async def resume(
        self,
        ctx: RunContext,
    ) -> StateTransition:
        """Resume a paused run.

        Args:
            ctx: The run context

        Returns:
            StateTransition record
        """
        # First transition to resuming
        await self.transition(ctx, RunState.RESUMING, reason="Resuming run")
        # Then to executing
        return await self.transition(ctx, RunState.EXECUTING, reason="Resumed")

    def delete_run(self, run_id: str) -> bool:
        """Delete a run (only if terminal).

        Args:
            run_id: The run ID

        Returns:
            True if deleted
        """
        ctx = self._runs.get(run_id)
        if ctx is None:
            return False

        if not ctx.is_terminal:
            raise StateError(
                "Cannot delete non-terminal run",
                context={"run_id": run_id, "state": ctx.state.value},
            )

        del self._runs[run_id]
        return True
