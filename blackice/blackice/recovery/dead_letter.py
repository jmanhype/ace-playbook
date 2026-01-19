"""Dead letter queue for BLACKICE 3.0.

Handles tasks that fail after maximum retries:
- Stores failed tasks for later analysis
- Supports manual retry requests
- Tracks failure reasons and history
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

from blackice.primitives.types import RunId


@dataclass
class DeadLetterEntry:
    """An entry in the dead letter queue.

    Attributes:
        id: Unique entry identifier
        run_id: Run this task belonged to
        task_name: Name of the failed task
        reason: Why task was moved to dead letter
        last_error: Last error message
        attempts: Number of attempts made
        retry_requested: Whether manual retry was requested
        created_at: When entry was created
        metadata: Optional additional data
    """

    id: str
    run_id: RunId
    task_name: str
    reason: str
    last_error: str
    attempts: int
    retry_requested: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": self.id,
            "run_id": str(self.run_id),  # Convert UUID to string
            "task_name": self.task_name,
            "reason": self.reason,
            "last_error": self.last_error,
            "attempts": self.attempts,
            "retry_requested": self.retry_requested,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeadLetterEntry:
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            task_name=data["task_name"],
            reason=data["reason"],
            last_error=data["last_error"],
            attempts=data["attempts"],
            retry_requested=data.get("retry_requested", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class DeadLetterQueue:
    """Queue for tasks that failed after maximum retries.

    Provides:
    - Storage for failed tasks
    - Query by run or globally
    - Manual retry marking
    - Entry removal after processing
    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the dead letter queue.

        Args:
            storage_dir: Directory for storing dead letter entries
        """
        self.storage_dir = storage_dir

    async def _ensure_dir(self) -> None:
        """Ensure storage directory exists."""
        await aiofiles.os.makedirs(self.storage_dir, exist_ok=True)

    def _get_entry_file(self, entry_id: str) -> Path:
        """Get the file path for an entry."""
        return self.storage_dir / f"{entry_id}.json"

    async def add(
        self,
        run_id: RunId,
        task_name: str,
        reason: str,
        last_error: str,
        attempts: int,
        metadata: dict[str, Any] | None = None,
    ) -> DeadLetterEntry:
        """Add a task to the dead letter queue.

        Args:
            run_id: Run the task belongs to
            task_name: Name of the failed task
            reason: Why task was moved here
            last_error: Last error message
            attempts: Number of attempts made
            metadata: Optional additional data

        Returns:
            Created DeadLetterEntry
        """
        await self._ensure_dir()

        entry = DeadLetterEntry(
            id=str(uuid4()),
            run_id=run_id,
            task_name=task_name,
            reason=reason,
            last_error=last_error,
            attempts=attempts,
            metadata=metadata or {},
        )

        entry_file = self._get_entry_file(entry.id)
        async with aiofiles.open(entry_file, "w") as f:
            await f.write(json.dumps(entry.to_dict(), indent=2))

        return entry

    async def list_entries(
        self,
        run_id: RunId | None = None,
    ) -> list[DeadLetterEntry]:
        """List dead letter entries.

        Args:
            run_id: Optional filter by run ID

        Returns:
            List of DeadLetterEntry objects
        """
        if not self.storage_dir.exists():
            return []

        entries: list[DeadLetterEntry] = []

        for entry_file in self.storage_dir.glob("*.json"):
            async with aiofiles.open(entry_file, "r") as f:
                content = await f.read()
            data = json.loads(content)
            entry = DeadLetterEntry.from_dict(data)

            if run_id is None or str(entry.run_id) == str(run_id):
                entries.append(entry)

        # Sort by creation time
        entries.sort(key=lambda e: e.created_at)
        return entries

    async def get_entry(self, entry_id: str) -> DeadLetterEntry | None:
        """Get a specific entry by ID.

        Args:
            entry_id: Entry ID to retrieve

        Returns:
            DeadLetterEntry or None if not found
        """
        entry_file = self._get_entry_file(entry_id)
        if not entry_file.exists():
            return None

        async with aiofiles.open(entry_file, "r") as f:
            content = await f.read()
        data = json.loads(content)
        return DeadLetterEntry.from_dict(data)

    async def mark_for_retry(self, entry_id: str) -> bool:
        """Mark an entry for manual retry.

        Args:
            entry_id: Entry to mark

        Returns:
            True if marked, False if not found
        """
        entry = await self.get_entry(entry_id)
        if entry is None:
            return False

        entry.retry_requested = True

        entry_file = self._get_entry_file(entry_id)
        async with aiofiles.open(entry_file, "w") as f:
            await f.write(json.dumps(entry.to_dict(), indent=2))

        return True

    async def remove(self, entry_id: str) -> bool:
        """Remove an entry from the queue.

        Args:
            entry_id: Entry to remove

        Returns:
            True if removed, False if not found
        """
        entry_file = self._get_entry_file(entry_id)
        if not entry_file.exists():
            return False

        await aiofiles.os.remove(entry_file)
        return True

    async def clear_run(self, run_id: RunId) -> int:
        """Clear all entries for a run.

        Args:
            run_id: Run to clear entries for

        Returns:
            Number of entries removed
        """
        entries = await self.list_entries(run_id)
        count = 0

        for entry in entries:
            if await self.remove(entry.id):
                count += 1

        return count

    async def get_retry_requested(self) -> list[DeadLetterEntry]:
        """Get entries that have been marked for retry.

        Returns:
            List of entries with retry_requested=True
        """
        all_entries = await self.list_entries()
        return [e for e in all_entries if e.retry_requested]

    async def count(self, run_id: RunId | None = None) -> int:
        """Count entries in the queue.

        Args:
            run_id: Optional filter by run ID

        Returns:
            Number of entries
        """
        entries = await self.list_entries(run_id)
        return len(entries)
