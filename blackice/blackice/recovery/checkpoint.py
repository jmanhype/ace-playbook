"""Checkpoint management for BLACKICE 3.0.

Provides crash recovery checkpoints that capture:
- Current event sequence
- Completed tasks
- In-flight task state
- Workspace snapshots
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import aiofiles.os

from blackice.persistence.event_store import EventStore
from blackice.primitives.types import EventType, RunId


@dataclass
class Checkpoint:
    """A recovery checkpoint capturing run state.

    Attributes:
        id: Unique checkpoint identifier
        run_id: Run this checkpoint belongs to
        event_sequence: Sequence number of last processed event
        completed_tasks: List of completed task names
        in_flight_tasks: Tasks started but not completed
        created_at: When checkpoint was created
        metadata: Optional additional data
    """

    id: str
    run_id: RunId
    event_sequence: int
    completed_tasks: list[str] = field(default_factory=list)
    in_flight_tasks: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "id": self.id,
            "run_id": str(self.run_id),  # Convert UUID to string
            "event_sequence": self.event_sequence,
            "completed_tasks": self.completed_tasks,
            "in_flight_tasks": self.in_flight_tasks,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create checkpoint from dictionary."""
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            event_sequence=data["event_sequence"],
            completed_tasks=data.get("completed_tasks", []),
            in_flight_tasks=data.get("in_flight_tasks", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """Manages crash recovery checkpoints.

    Checkpoints capture run state at specific points, enabling:
    - Fast recovery without full event replay
    - Skip completed work on resume
    - Identify in-flight tasks for retry
    """

    def __init__(self, event_store: EventStore, checkpoint_dir: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            event_store: Event store for reading events
            checkpoint_dir: Directory for storing checkpoints
        """
        self.event_store = event_store
        self.checkpoint_dir = checkpoint_dir

    async def _ensure_dir(self, run_id: RunId) -> Path:
        """Ensure checkpoint directory exists for run."""
        run_dir = self.checkpoint_dir / str(run_id)
        await aiofiles.os.makedirs(run_dir, exist_ok=True)
        return run_dir

    async def create_checkpoint(
        self,
        run_id: RunId,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint for a run.

        Analyzes events to determine:
        - Last processed event sequence
        - Which tasks are completed
        - Which tasks are in-flight

        Args:
            run_id: Run to checkpoint
            metadata: Optional additional data to include

        Returns:
            Created Checkpoint
        """
        events = await self.event_store.get_events(run_id)

        # Analyze events for task state
        completed_tasks: list[str] = []
        in_flight_tasks: set[str] = set()
        last_sequence = -1

        for event in events:
            last_sequence = event.sequence

            if event.type == EventType.TASK_STARTED:
                task_name = event.payload.get("task_name", "")
                if task_name:
                    in_flight_tasks.add(task_name)

            elif event.type == EventType.TASK_COMPLETED:
                task_name = event.payload.get("task_name", "")
                if task_name:
                    completed_tasks.append(task_name)
                    in_flight_tasks.discard(task_name)

            elif event.type == EventType.TASK_FAILED:
                task_name = event.payload.get("task_name", "")
                if task_name:
                    in_flight_tasks.discard(task_name)

        checkpoint = Checkpoint(
            id=str(uuid4()),
            run_id=run_id,
            event_sequence=last_sequence,
            completed_tasks=completed_tasks,
            in_flight_tasks=list(in_flight_tasks),
            metadata=metadata or {},
        )

        # Persist checkpoint
        run_dir = await self._ensure_dir(run_id)
        checkpoint_file = run_dir / f"checkpoint-{checkpoint.id}.json"

        async with aiofiles.open(checkpoint_file, "w") as f:
            await f.write(json.dumps(checkpoint.to_dict(), indent=2))

        # Also write an event for the checkpoint
        from blackice.schemas.event import EventPayloads

        await self.event_store.append(
            run_id,
            EventType.CHECKPOINT_CREATED,
            EventPayloads.checkpoint_created(checkpoint.id, last_sequence),
        )

        return checkpoint

    async def get_latest_checkpoint(self, run_id: RunId) -> Checkpoint | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: Run to get checkpoint for

        Returns:
            Latest Checkpoint or None if no checkpoints exist
        """
        run_dir = self.checkpoint_dir / str(run_id)
        if not run_dir.exists():
            return None

        # Find all checkpoints and sort by modification time
        checkpoint_files = list(run_dir.glob("checkpoint-*.json"))
        if not checkpoint_files:
            return None

        # Sort by modification time (most recent last)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime)

        # Load the latest (most recently modified)
        latest_file = checkpoint_files[-1]
        async with aiofiles.open(latest_file, "r") as f:
            content = await f.read()

        data = json.loads(content)
        return Checkpoint.from_dict(data)

    async def list_checkpoints(self, run_id: RunId) -> list[Checkpoint]:
        """List all checkpoints for a run.

        Args:
            run_id: Run to list checkpoints for

        Returns:
            List of Checkpoints in creation order
        """
        run_dir = self.checkpoint_dir / str(run_id)
        if not run_dir.exists():
            return []

        checkpoints: list[Checkpoint] = []
        for checkpoint_file in sorted(run_dir.glob("checkpoint-*.json")):
            async with aiofiles.open(checkpoint_file, "r") as f:
                content = await f.read()
            data = json.loads(content)
            checkpoints.append(Checkpoint.from_dict(data))

        return checkpoints

    async def delete_checkpoints(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: Run to delete checkpoints for

        Returns:
            Number of checkpoints deleted
        """
        run_dir = self.checkpoint_dir / str(run_id)
        if not run_dir.exists():
            return 0

        count = 0
        for checkpoint_file in run_dir.glob("checkpoint-*.json"):
            await aiofiles.os.remove(checkpoint_file)
            count += 1

        # Remove directory if empty
        try:
            run_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other error

        return count
