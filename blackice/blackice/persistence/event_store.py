"""Event store with JSONL persistence for BLACKICE 3.0.

Provides durable event storage with:
- JSONL (JSON Lines) format for append-only storage
- Hash chain for tamper detection
- Per-run file isolation
- Event retrieval and filtering
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from blackice.primitives.types import EventType, RunId, new_event_id
from blackice.schemas.event import Event


@dataclass
class EventStoreConfig:
    """Configuration for the event store."""

    storage_dir: Path = field(default_factory=lambda: Path.home() / ".blackice" / "events")
    file_extension: str = ".jsonl"
    sync_on_write: bool = True
    max_events_per_file: int = 10000


class EventStore:
    """Persistent event store with JSONL format.

    Events are stored in per-run JSONL files with:
    - One JSON object per line
    - Hash chain linking events
    - Sequence numbers for ordering
    - Atomic append operations

    Supports:
    - Append new events
    - Retrieve events by run/type/sequence
    - Verify hash chain integrity
    - List all runs
    """

    def __init__(self, config: EventStoreConfig | None = None) -> None:
        """Initialize the event store.

        Args:
            config: Store configuration, uses defaults if not provided
        """
        self.config = config or EventStoreConfig()
        self._locks: dict[str, asyncio.Lock] = {}
        self._sequence_cache: dict[str, int] = {}
        self._hash_cache: dict[str, str | None] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the event store, creating directories if needed."""
        if self._initialized:
            return

        await aiofiles.os.makedirs(self.config.storage_dir, exist_ok=True)
        self._initialized = True

    async def close(self) -> None:
        """Close the event store and release resources."""
        self._locks.clear()
        self._sequence_cache.clear()
        self._hash_cache.clear()
        self._initialized = False

    def _get_lock(self, run_id: RunId) -> asyncio.Lock:
        """Get or create a lock for a specific run."""
        if run_id not in self._locks:
            self._locks[run_id] = asyncio.Lock()
        return self._locks[run_id]

    def _get_event_file(self, run_id: RunId) -> Path:
        """Get the event file path for a run."""
        return self.config.storage_dir / f"{run_id}{self.config.file_extension}"

    async def _load_state(self, run_id: RunId) -> None:
        """Load sequence and hash state from existing events."""
        if run_id in self._sequence_cache:
            return

        event_file = self._get_event_file(run_id)
        if not event_file.exists():
            self._sequence_cache[run_id] = 0
            self._hash_cache[run_id] = None
            return

        # Read last event to get current state
        last_event = await self._read_last_event(event_file)
        if last_event:
            self._sequence_cache[run_id] = last_event.sequence + 1
            self._hash_cache[run_id] = last_event.hash
        else:
            self._sequence_cache[run_id] = 0
            self._hash_cache[run_id] = None

    async def _read_last_event(self, event_file: Path) -> Event | None:
        """Read the last event from a file."""
        if not event_file.exists():
            return None

        async with aiofiles.open(event_file, "r") as f:
            content = await f.read()

        lines = content.strip().split("\n")
        if not lines or not lines[-1]:
            return None

        data = json.loads(lines[-1])
        return self._dict_to_event(data)

    def _event_to_dict(self, event: Event) -> dict[str, Any]:
        """Convert an event to a dictionary for JSON serialization."""
        # Convert timestamp to ISO string
        if hasattr(event.timestamp, "value"):
            timestamp_str = event.timestamp.value.isoformat()
        elif hasattr(event.timestamp, "isoformat"):
            timestamp_str = event.timestamp.isoformat()
        else:
            timestamp_str = str(event.timestamp)

        return {
            "id": str(event.id),
            "run_id": str(event.run_id),
            "type": event.type.value,
            "payload": event.payload,
            "timestamp": timestamp_str,
            "correlation_id": event.correlation_id,
            "task_id": str(event.task_id) if event.task_id else None,
            "agent_id": event.agent_id,
            "sequence": event.sequence,
            "previous_hash": event.previous_hash,
            "hash": event.hash,
        }

    def _dict_to_event(self, data: dict[str, Any]) -> Event:
        """Convert a dictionary to an Event."""
        from datetime import datetime, timezone

        from blackice.primitives.types import Timestamp

        # Convert type string back to enum
        event_type = EventType(data["type"])

        # Parse timestamp
        timestamp_str = data.get("timestamp", "")
        if isinstance(timestamp_str, str) and timestamp_str:
            # Parse the ISO format timestamp
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            timestamp = Timestamp(value=dt)
        else:
            timestamp = Timestamp.now()

        return Event(
            id=data["id"],
            run_id=data["run_id"],
            type=event_type,
            payload=data.get("payload", {}),
            timestamp=timestamp,
            correlation_id=data.get("correlation_id"),
            task_id=data.get("task_id"),
            agent_id=data.get("agent_id"),
            sequence=data["sequence"],
            previous_hash=data.get("previous_hash"),
        )

    async def append(
        self,
        run_id: RunId,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> Event:
        """Append a new event to the store.

        Args:
            run_id: Run this event belongs to
            event_type: Type of event
            payload: Event-specific data
            correlation_id: Optional correlation ID for tracing
            task_id: Optional task ID
            agent_id: Optional agent ID

        Returns:
            The created Event
        """
        async with self._get_lock(run_id):
            await self._load_state(run_id)

            sequence = self._sequence_cache[run_id]
            previous_hash = self._hash_cache[run_id]

            event = Event(
                id=new_event_id(),
                run_id=run_id,
                type=event_type,
                payload=payload,
                sequence=sequence,
                previous_hash=previous_hash,
                correlation_id=correlation_id,
                task_id=task_id,
                agent_id=agent_id,
            )

            # Write to file
            event_file = self._get_event_file(run_id)
            event_dict = self._event_to_dict(event)
            line = json.dumps(event_dict, default=str) + "\n"

            async with aiofiles.open(event_file, "a") as f:
                await f.write(line)
                if self.config.sync_on_write:
                    await f.flush()

            # Update cache
            self._sequence_cache[run_id] = sequence + 1
            self._hash_cache[run_id] = event.hash

            return event

    async def get_events(
        self,
        run_id: RunId,
        *,
        event_type: EventType | None = None,
        since_sequence: int | None = None,
    ) -> list[Event]:
        """Get events for a run.

        Args:
            run_id: Run to get events for
            event_type: Optional filter by event type
            since_sequence: Optional filter to events after this sequence

        Returns:
            List of matching events in sequence order
        """
        event_file = self._get_event_file(run_id)
        if not event_file.exists():
            return []

        events: list[Event] = []

        async with aiofiles.open(event_file, "r") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                event = self._dict_to_event(data)

                # Apply filters
                if since_sequence is not None and event.sequence < since_sequence:
                    continue
                if event_type is not None and event.type != event_type:
                    continue

                events.append(event)

        return events

    async def get_latest_event(self, run_id: RunId) -> Event | None:
        """Get the most recent event for a run.

        Args:
            run_id: Run to get latest event for

        Returns:
            The most recent event, or None if no events exist
        """
        event_file = self._get_event_file(run_id)
        return await self._read_last_event(event_file)

    async def verify_integrity(self, run_id: RunId) -> tuple[bool, str | None]:
        """Verify the hash chain integrity for a run.

        Args:
            run_id: Run to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        events = await self.get_events(run_id)

        if not events:
            return True, None

        previous_hash: str | None = None

        for i, event in enumerate(events):
            # Check sequence
            if event.sequence != i:
                return False, f"Sequence mismatch at index {i}: expected {i}, got {event.sequence}"

            # Check hash chain
            if event.previous_hash != previous_hash:
                return False, f"Hash chain broken at sequence {i}"

            previous_hash = event.hash

        return True, None

    async def list_runs(self) -> list[RunId]:
        """List all runs with stored events.

        Returns:
            List of run IDs
        """
        runs: list[RunId] = []

        if not self.config.storage_dir.exists():
            return runs

        for file_path in self.config.storage_dir.iterdir():
            if file_path.suffix == self.config.file_extension:
                run_id = file_path.stem
                runs.append(run_id)

        return runs

    async def get_event_count(self, run_id: RunId) -> int:
        """Get the number of events for a run.

        Args:
            run_id: Run to count events for

        Returns:
            Number of events
        """
        event_file = self._get_event_file(run_id)
        if not event_file.exists():
            return 0

        count = 0
        async with aiofiles.open(event_file, "r") as f:
            async for line in f:
                if line.strip():
                    count += 1

        return count

    async def delete_run(self, run_id: RunId) -> bool:
        """Delete all events for a run.

        Args:
            run_id: Run to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._get_lock(run_id):
            event_file = self._get_event_file(run_id)
            if not event_file.exists():
                return False

            await aiofiles.os.remove(event_file)

            # Clear cache
            self._sequence_cache.pop(run_id, None)
            self._hash_cache.pop(run_id, None)

            return True
