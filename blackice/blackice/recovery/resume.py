"""Resume management for BLACKICE 3.0.

Provides crash recovery resume functionality:
- Identifies completed tasks to skip
- Identifies in-flight tasks for retry with new attempts
- Manages idempotency keys for external effects
- Determines optimal resume point
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blackice.persistence.event_store import EventStore
from blackice.primitives.types import EventType, RunId, TaskId
from blackice.recovery.checkpoint import Checkpoint, CheckpointManager


@dataclass
class ResumeState:
    """State information for resuming a run.

    Attributes:
        run_id: Run being resumed
        last_checkpoint_id: ID of checkpoint being resumed from
        resume_from_sequence: Event sequence to resume from
        completed_tasks: Tasks that completed successfully
        in_flight_tasks: Tasks that were started but not completed
        task_attempts: Map of task to last attempt number
        used_idempotency_keys: Set of idempotency keys already used
    """

    run_id: RunId
    last_checkpoint_id: str | None = None
    resume_from_sequence: int = 0
    completed_tasks: set[str] = field(default_factory=set)
    in_flight_tasks: set[str] = field(default_factory=set)
    task_attempts: dict[str, int] = field(default_factory=dict)
    used_idempotency_keys: set[str] = field(default_factory=set)

    def should_skip_task(self, task_name: str) -> bool:
        """Check if a task should be skipped (already completed)."""
        return task_name in self.completed_tasks

    def get_next_attempt(self, task_name: str) -> int:
        """Get the next attempt number for a task."""
        return self.task_attempts.get(task_name, 0) + 1

    def is_idempotent_key_used(self, key: str) -> bool:
        """Check if an idempotency key has been used."""
        return key in self.used_idempotency_keys


class ResumeManager:
    """Manages run resume operations.

    Enables:
    - Resume crashed runs without repeating work
    - Retry failed tasks with new attempt IDs
    - Skip effects with used idempotency keys
    """

    def __init__(self, event_store: EventStore, checkpoint_dir: Path) -> None:
        """Initialize the resume manager.

        Args:
            event_store: Event store for reading events
            checkpoint_dir: Directory for checkpoints
        """
        self.event_store = event_store
        self.checkpoint_dir = checkpoint_dir
        self._checkpoint_mgr = CheckpointManager(event_store, checkpoint_dir)

    async def get_resume_state(self, run_id: RunId) -> ResumeState:
        """Get the resume state for a run.

        Analyzes events and checkpoints to determine:
        - What work was completed
        - What was in progress
        - What idempotency keys were used

        Args:
            run_id: Run to get resume state for

        Returns:
            ResumeState with all resume information
        """
        state = ResumeState(run_id=run_id)

        # Try to load from checkpoint first
        checkpoint = await self._checkpoint_mgr.get_latest_checkpoint(run_id)
        if checkpoint:
            state.last_checkpoint_id = checkpoint.id
            state.resume_from_sequence = checkpoint.event_sequence
            state.completed_tasks = set(checkpoint.completed_tasks)
            state.in_flight_tasks = set(checkpoint.in_flight_tasks)

        # Always replay events from beginning to get accurate state
        # (checkpoint may be stale)
        events = await self.event_store.get_events(run_id)

        for event in events:
            if event.type == EventType.TASK_STARTED:
                task_name = event.payload.get("task_name", "")
                attempt = event.payload.get("attempt", 1)
                if task_name:
                    state.in_flight_tasks.add(task_name)
                    state.task_attempts[task_name] = attempt

            elif event.type == EventType.TASK_COMPLETED:
                task_name = event.payload.get("task_name", "")
                if task_name:
                    state.completed_tasks.add(task_name)
                    state.in_flight_tasks.discard(task_name)

            elif event.type == EventType.TASK_FAILED:
                task_name = event.payload.get("task_name", "")
                if task_name:
                    state.in_flight_tasks.discard(task_name)

            elif event.type == EventType.COMMAND_EXECUTED:
                idem_key = event.payload.get("idempotency_key")
                if idem_key:
                    state.used_idempotency_keys.add(idem_key)

        return state

    def generate_idempotency_key(
        self,
        run_id: RunId,
        task_id: TaskId,
        attempt: int,
        effect_type: str = "default",
    ) -> str:
        """Generate a unique idempotency key for an effect.

        Keys are deterministic based on:
        - Run ID
        - Task ID
        - Attempt number
        - Effect type

        This ensures:
        - Same effect in same attempt → same key (idempotent)
        - Same effect in different attempt → different key (retryable)

        Args:
            run_id: Current run ID
            task_id: Current task ID
            attempt: Current attempt number
            effect_type: Type of effect (e.g., "file_write", "api_call")

        Returns:
            Unique idempotency key
        """
        components = [str(run_id), str(task_id), str(attempt), effect_type]
        combined = ":".join(components)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"{task_id}-{attempt}-{effect_type}-{hash_value}"

    async def can_resume(self, run_id: RunId) -> tuple[bool, str]:
        """Check if a run can be resumed.

        Args:
            run_id: Run to check

        Returns:
            Tuple of (can_resume, reason)
        """
        events = await self.event_store.get_events(run_id)

        if not events:
            return False, "No events found for run"

        # Check if run is already completed
        for event in events:
            if event.type == EventType.RUN_COMPLETED:
                return False, "Run already completed successfully"
            if event.type == EventType.RUN_FAILED:
                # Failed runs can be resumed
                pass

        # Find the last event to determine state
        latest = events[-1] if events else None
        if latest and latest.type == EventType.RUN_STARTED:
            # Just started, nothing to resume
            return False, "Run just started, nothing to resume"

        return True, "Run can be resumed"

    async def get_pending_tasks(
        self,
        run_id: RunId,
        all_tasks: list[str],
    ) -> list[tuple[str, int]]:
        """Get tasks that still need to run.

        Args:
            run_id: Run to check
            all_tasks: Complete list of tasks for the run

        Returns:
            List of (task_name, attempt_number) for pending tasks
        """
        state = await self.get_resume_state(run_id)
        pending: list[tuple[str, int]] = []

        for task_name in all_tasks:
            if task_name in state.completed_tasks:
                continue  # Skip completed

            attempt = state.get_next_attempt(task_name)
            pending.append((task_name, attempt))

        return pending

    async def create_resume_checkpoint(self, run_id: RunId) -> None:
        """Create a checkpoint at the current resume point.

        Args:
            run_id: Run to checkpoint
        """
        await self._checkpoint_mgr.create_checkpoint(
            run_id,
            metadata={"checkpoint_type": "resume"},
        )
