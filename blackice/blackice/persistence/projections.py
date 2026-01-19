"""Event projections for BLACKICE 3.0.

Provides deterministic state reconstruction from events:
- RunProjection: Reconstruct run state from events
- TaskProjection: Reconstruct task state from events
- Projection: Base class for custom projections
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from blackice.primitives.types import EventType, RunId, TaskId
from blackice.schemas.event import Event


class RunStatus(str, Enum):
    """Run status states."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Task status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


T = TypeVar("T")


class Projection(ABC, Generic[T]):
    """Base class for event projections.

    Projections fold events into a state representation,
    enabling deterministic state reconstruction.
    """

    @abstractmethod
    def apply(self, event: Event) -> None:
        """Apply an event to update state.

        Args:
            event: Event to apply
        """
        pass

    @abstractmethod
    def get_state(self) -> T:
        """Get the current projected state.

        Returns:
            Current state
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        pass


@dataclass
class RunState:
    """Projected state of a run.

    Attributes:
        run_id: Run identifier
        status: Current status
        vision: Original vision description
        edition: BLACKICE edition (lite, core, enterprise)
        config: Run configuration
        started_at: When run started
        completed_at: When run completed (if finished)
        error_message: Error message (if failed)
        event_count: Number of events processed
        last_sequence: Last event sequence number
        completed_tasks: List of completed task names
        failed_tasks: List of failed task names
        in_flight_tasks: Currently running tasks
        checkpoints: List of checkpoint IDs
        metadata: Additional metadata
    """

    run_id: RunId
    status: RunStatus = RunStatus.PENDING
    vision: str = ""
    edition: str = "lite"
    config: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    event_count: int = 0
    last_sequence: int = -1
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    in_flight_tasks: set[str] = field(default_factory=set)
    checkpoints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RunProjection(Projection[RunState]):
    """Projects events into run state.

    Reconstructs the complete run state from the event stream,
    enabling crash recovery and state inspection.
    """

    def __init__(self, run_id: RunId) -> None:
        """Initialize the projection.

        Args:
            run_id: Run to project
        """
        self._state = RunState(run_id=run_id)

    def apply(self, event: Event) -> None:
        """Apply an event to update run state.

        Args:
            event: Event to apply
        """
        self._state.event_count += 1
        self._state.last_sequence = event.sequence

        handler = getattr(self, f"_handle_{event.type.value}", None)
        if handler:
            handler(event)

    def get_state(self) -> RunState:
        """Get the current run state.

        Returns:
            Current RunState
        """
        return self._state

    def reset(self) -> None:
        """Reset to initial state."""
        run_id = self._state.run_id
        self._state = RunState(run_id=run_id)

    def _handle_run_started(self, event: Event) -> None:
        """Handle RUN_STARTED event."""
        self._state.status = RunStatus.PLANNING
        self._state.vision = event.payload.get("vision", "")
        self._state.edition = event.payload.get("edition", "lite")
        self._state.config = event.payload.get("config", {})
        # Parse timestamp from event
        if hasattr(event.timestamp, "value"):
            self._state.started_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            self._state.started_at = event.timestamp

    def _handle_run_completed(self, event: Event) -> None:
        """Handle RUN_COMPLETED event."""
        self._state.status = RunStatus.COMPLETED
        if hasattr(event.timestamp, "value"):
            self._state.completed_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            self._state.completed_at = event.timestamp

    def _handle_run_failed(self, event: Event) -> None:
        """Handle RUN_FAILED event."""
        self._state.status = RunStatus.FAILED
        self._state.error_message = event.payload.get("error", "Unknown error")
        if hasattr(event.timestamp, "value"):
            self._state.completed_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            self._state.completed_at = event.timestamp

    def _handle_phase_started(self, event: Event) -> None:
        """Handle PHASE_STARTED event."""
        phase = event.payload.get("phase", "")
        if phase == "plan":
            self._state.status = RunStatus.PLANNING
        elif phase in ("implement", "execute"):
            self._state.status = RunStatus.EXECUTING
        elif phase in ("verify", "test"):
            self._state.status = RunStatus.VERIFYING

    def _handle_task_started(self, event: Event) -> None:
        """Handle TASK_STARTED event."""
        task_name = event.payload.get("task_name", "")
        if task_name:
            self._state.in_flight_tasks.add(task_name)

    def _handle_task_completed(self, event: Event) -> None:
        """Handle TASK_COMPLETED event."""
        task_name = event.payload.get("task_name", "")
        if task_name:
            self._state.in_flight_tasks.discard(task_name)
            if task_name not in self._state.completed_tasks:
                self._state.completed_tasks.append(task_name)

    def _handle_task_failed(self, event: Event) -> None:
        """Handle TASK_FAILED event."""
        task_name = event.payload.get("task_name", "")
        if task_name:
            self._state.in_flight_tasks.discard(task_name)
            if task_name not in self._state.failed_tasks:
                self._state.failed_tasks.append(task_name)

    def _handle_checkpoint_created(self, event: Event) -> None:
        """Handle CHECKPOINT_CREATED event."""
        checkpoint_id = event.payload.get("checkpoint_id", "")
        if checkpoint_id:
            self._state.checkpoints.append(checkpoint_id)


@dataclass
class TaskState:
    """Projected state of a task.

    Attributes:
        task_id: Task identifier (optional, may use name)
        task_name: Task name
        run_id: Parent run ID
        status: Current status
        attempt: Current attempt number
        started_at: When task started
        completed_at: When task completed
        duration_seconds: Execution duration
        error_message: Error message (if failed)
        output: Task output/result
        metadata: Additional metadata
    """

    task_name: str
    run_id: RunId
    task_id: TaskId | None = None
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskProjection(Projection[dict[str, TaskState]]):
    """Projects events into task states.

    Maintains state for all tasks in a run, enabling
    task-level recovery and status tracking.
    """

    def __init__(self, run_id: RunId) -> None:
        """Initialize the projection.

        Args:
            run_id: Run to project tasks for
        """
        self._run_id = run_id
        self._tasks: dict[str, TaskState] = {}

    def apply(self, event: Event) -> None:
        """Apply an event to update task states.

        Args:
            event: Event to apply
        """
        handler = getattr(self, f"_handle_{event.type.value}", None)
        if handler:
            handler(event)

    def get_state(self) -> dict[str, TaskState]:
        """Get all task states.

        Returns:
            Dictionary of task_name -> TaskState
        """
        return self._tasks.copy()

    def get_task(self, task_name: str) -> TaskState | None:
        """Get state for a specific task.

        Args:
            task_name: Task name to look up

        Returns:
            TaskState or None if not found
        """
        return self._tasks.get(task_name)

    def reset(self) -> None:
        """Reset to initial state."""
        self._tasks.clear()

    def _ensure_task(self, task_name: str) -> TaskState:
        """Ensure task state exists.

        Args:
            task_name: Task name

        Returns:
            TaskState for the task
        """
        if task_name not in self._tasks:
            self._tasks[task_name] = TaskState(
                task_name=task_name,
                run_id=self._run_id,
            )
        return self._tasks[task_name]

    def _handle_task_started(self, event: Event) -> None:
        """Handle TASK_STARTED event."""
        task_name = event.payload.get("task_name", "")
        if not task_name:
            return

        task = self._ensure_task(task_name)
        task.status = TaskStatus.RUNNING
        task.attempt = event.payload.get("attempt", 1)
        task.task_id = event.task_id

        if hasattr(event.timestamp, "value"):
            task.started_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            task.started_at = event.timestamp

    def _handle_task_completed(self, event: Event) -> None:
        """Handle TASK_COMPLETED event."""
        task_name = event.payload.get("task_name", "")
        if not task_name:
            return

        task = self._ensure_task(task_name)
        task.status = TaskStatus.COMPLETED
        task.duration_seconds = event.payload.get("duration_seconds", 0.0)
        task.output = event.payload.get("output", {})

        if hasattr(event.timestamp, "value"):
            task.completed_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            task.completed_at = event.timestamp

    def _handle_task_failed(self, event: Event) -> None:
        """Handle TASK_FAILED event."""
        task_name = event.payload.get("task_name", "")
        if not task_name:
            return

        task = self._ensure_task(task_name)
        task.status = TaskStatus.FAILED
        task.error_message = event.payload.get("error", "Unknown error")

        if hasattr(event.timestamp, "value"):
            task.completed_at = event.timestamp.value
        elif hasattr(event.timestamp, "isoformat"):
            task.completed_at = event.timestamp

    def _handle_task_skipped(self, event: Event) -> None:
        """Handle TASK_SKIPPED event."""
        task_name = event.payload.get("task_name", "")
        if not task_name:
            return

        task = self._ensure_task(task_name)
        task.status = TaskStatus.SKIPPED
        task.metadata["skip_reason"] = event.payload.get("reason", "")


def reconstruct_run_state(events: list[Event], run_id: RunId) -> RunState:
    """Reconstruct run state from events.

    Convenience function that creates a projection and
    applies all events.

    Args:
        events: List of events in sequence order
        run_id: Run ID to reconstruct

    Returns:
        Reconstructed RunState
    """
    projection = RunProjection(run_id)
    for event in events:
        projection.apply(event)
    return projection.get_state()


def reconstruct_task_states(
    events: list[Event],
    run_id: RunId,
) -> dict[str, TaskState]:
    """Reconstruct all task states from events.

    Convenience function that creates a projection and
    applies all events.

    Args:
        events: List of events in sequence order
        run_id: Run ID to reconstruct

    Returns:
        Dictionary of task_name -> TaskState
    """
    projection = TaskProjection(run_id)
    for event in events:
        projection.apply(event)
    return projection.get_state()
