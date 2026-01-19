"""Task schema for BLACKICE 3.0.

Tasks represent individual units of work within a Run, following
the tree structure of the execution plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import (
    RunId,
    TaskId,
    TaskStatus,
    Timestamp,
    new_task_id,
)


class TaskFiles(BaseModel):
    """File paths associated with a task for checkpointing."""

    plan_file: Path | None = Field(default=None, description="Task plan/spec file")
    state_file: Path | None = Field(default=None, description="Serialized task state")
    notes_file: Path | None = Field(default=None, description="Agent notes/reflections")
    output_file: Path | None = Field(default=None, description="Task output/artifacts")


class Task(BaseModel):
    """An individual task within a BLACKICE run.

    Tasks are the atomic units of work that make up a run. They form
    a dependency graph and can be checkpointed for recovery.

    Attributes:
        id: Unique identifier for this task
        run_id: Parent run this task belongs to
        name: Human-readable task name
        description: Detailed task description
        status: Current task status
        attempt: Current attempt number (for retries)
        max_attempts: Maximum retry attempts allowed
        dependencies: Task IDs this task depends on
        files: Associated file paths for checkpointing
        idempotency_key: Key for detecting duplicate executions
    """

    id: TaskId = Field(default_factory=new_task_id)
    run_id: RunId
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)

    # Status tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)

    # Timing
    created_at: Timestamp = Field(default_factory=Timestamp.now)
    started_at: Timestamp | None = Field(default=None)
    completed_at: Timestamp | None = Field(default=None)

    # Dependencies
    dependencies: list[TaskId] = Field(default_factory=list)
    blocked_by: list[TaskId] = Field(default_factory=list)

    # Checkpointing
    files: TaskFiles = Field(default_factory=TaskFiles)
    idempotency_key: str | None = Field(default=None, description="Prevent duplicate execution")

    # Execution context
    agent_id: str | None = Field(default=None, description="Assigned agent")
    parent_task_id: TaskId | None = Field(default=None, description="Parent task (if subtask)")

    # Results
    result: Any = Field(default=None, description="Task execution result")
    error: str | None = Field(default=None, description="Error message if failed")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (dependencies met)."""
        return self.status == TaskStatus.PENDING and len(self.blocked_by) == 0

    @property
    def is_terminal(self) -> bool:
        """Check if task has reached a terminal state."""
        return self.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED)

    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.attempt < self.max_attempts and self.status == TaskStatus.FAILED

    def start(self) -> Task:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = Timestamp.now()
        self.attempt += 1
        return self

    def block(self, blocker_id: TaskId) -> Task:
        """Mark task as blocked by another task."""
        self.status = TaskStatus.BLOCKED
        if blocker_id not in self.blocked_by:
            self.blocked_by.append(blocker_id)
        return self

    def unblock(self, blocker_id: TaskId) -> Task:
        """Remove a blocker from this task."""
        if blocker_id in self.blocked_by:
            self.blocked_by.remove(blocker_id)
        if len(self.blocked_by) == 0 and self.status == TaskStatus.BLOCKED:
            self.status = TaskStatus.PENDING
        return self

    def succeed(self, result: Any = None) -> Task:
        """Mark task as successfully completed."""
        self.status = TaskStatus.SUCCEEDED
        self.completed_at = Timestamp.now()
        self.result = result
        return self

    def fail(self, error: str) -> Task:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = Timestamp.now()
        self.error = error
        return self

    def skip(self, reason: str = "Skipped") -> Task:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.completed_at = Timestamp.now()
        self.metadata["skip_reason"] = reason
        return self

    def reset_for_retry(self) -> Task:
        """Reset task state for retry attempt."""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        return self


class TaskGraph(BaseModel):
    """A directed acyclic graph of tasks for a run.

    Manages task dependencies and execution ordering.
    """

    run_id: RunId
    tasks: dict[TaskId, Task] = Field(default_factory=dict)
    execution_order: list[TaskId] = Field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        self.tasks[task.id] = task

    def get_ready_tasks(self) -> list[Task]:
        """Get all tasks that are ready to execute."""
        return [task for task in self.tasks.values() if task.is_ready]

    def get_blocked_tasks(self) -> list[Task]:
        """Get all blocked tasks."""
        return [
            task for task in self.tasks.values() if task.status == TaskStatus.BLOCKED
        ]

    def mark_complete(self, task_id: TaskId) -> list[Task]:
        """Mark a task complete and unblock dependents.

        Returns list of newly unblocked tasks.
        """
        unblocked: list[Task] = []
        for task in self.tasks.values():
            if task_id in task.blocked_by:
                task.unblock(task_id)
                if task.status == TaskStatus.PENDING:
                    unblocked.append(task)
        return unblocked

    @property
    def all_complete(self) -> bool:
        """Check if all tasks have completed."""
        return all(task.is_terminal for task in self.tasks.values())

    @property
    def has_failures(self) -> bool:
        """Check if any tasks have failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())
