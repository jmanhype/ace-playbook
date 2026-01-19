"""Idempotency key management for BLACKICE 3.0.

Provides idempotency handling for external effects:
- Key generation (deterministic, unique per attempt)
- Key tracking and persistence
- Enforcement of idempotency guarantees
- Integration with execution providers
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

import aiofiles
import aiofiles.os

from blackice.primitives.types import RunId, TaskId


class EffectType(str, Enum):
    """Types of external effects."""

    FILE_WRITE = "file_write"  # Writing to files
    FILE_DELETE = "file_delete"  # Deleting files
    COMMAND_EXEC = "command_exec"  # Command execution
    API_CALL = "api_call"  # External API calls
    DB_WRITE = "db_write"  # Database writes
    MESSAGE_SEND = "message_send"  # Sending messages
    OTHER = "other"  # Other effects


@dataclass
class IdempotencyRecord:
    """Record of an idempotent operation.

    Attributes:
        key: Unique idempotency key
        run_id: Run this operation belongs to
        task_id: Task this operation belongs to
        effect_type: Type of effect
        attempt: Attempt number when executed
        executed_at: When operation was executed
        result_hash: Hash of operation result (for verification)
        metadata: Additional metadata
    """

    key: str
    run_id: str
    task_id: str
    effect_type: EffectType
    attempt: int
    executed_at: datetime = field(default_factory=datetime.utcnow)
    result_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "effect_type": self.effect_type.value,
            "attempt": self.attempt,
            "executed_at": self.executed_at.isoformat(),
            "result_hash": self.result_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdempotencyRecord:
        """Create from dictionary."""
        return cls(
            key=data["key"],
            run_id=data["run_id"],
            task_id=data["task_id"],
            effect_type=EffectType(data["effect_type"]),
            attempt=data["attempt"],
            executed_at=datetime.fromisoformat(data["executed_at"]),
            result_hash=data.get("result_hash"),
            metadata=data.get("metadata", {}),
        )


class IdempotencyKeyGenerator:
    """Generates deterministic idempotency keys.

    Keys are unique per:
    - Run ID
    - Task ID
    - Attempt number
    - Effect type
    - Effect-specific parameters

    Same inputs always produce the same key, enabling
    reliable duplicate detection on resume.
    """

    @staticmethod
    def generate(
        run_id: str,
        task_id: str,
        attempt: int,
        effect_type: EffectType,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a deterministic idempotency key.

        Args:
            run_id: Run identifier
            task_id: Task identifier
            attempt: Attempt number
            effect_type: Type of effect
            params: Effect-specific parameters

        Returns:
            Unique, deterministic idempotency key
        """
        # Build key components
        components = [
            str(run_id),
            str(task_id),
            str(attempt),
            effect_type.value,
        ]

        # Add sorted params for determinism
        if params:
            param_str = json.dumps(params, sort_keys=True)
            components.append(param_str)

        # Hash the combined components
        combined = ":".join(components)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]

        # Return readable key format
        return f"{task_id[:8]}-{attempt}-{effect_type.value}-{hash_value}"

    @staticmethod
    def generate_for_command(
        run_id: str,
        task_id: str,
        attempt: int,
        command: str,
    ) -> str:
        """Generate key for command execution.

        Args:
            run_id: Run identifier
            task_id: Task identifier
            attempt: Attempt number
            command: Command being executed

        Returns:
            Idempotency key
        """
        return IdempotencyKeyGenerator.generate(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            effect_type=EffectType.COMMAND_EXEC,
            params={"command": command},
        )

    @staticmethod
    def generate_for_file_write(
        run_id: str,
        task_id: str,
        attempt: int,
        file_path: str,
        content_hash: str,
    ) -> str:
        """Generate key for file write.

        Args:
            run_id: Run identifier
            task_id: Task identifier
            attempt: Attempt number
            file_path: Path being written
            content_hash: Hash of content being written

        Returns:
            Idempotency key
        """
        return IdempotencyKeyGenerator.generate(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            effect_type=EffectType.FILE_WRITE,
            params={"path": file_path, "content_hash": content_hash},
        )


class IdempotencyStore:
    """Persistent storage for idempotency records.

    Tracks executed operations to enable:
    - Duplicate detection on resume
    - Skipping already-executed effects
    - Audit trail of external effects
    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the store.

        Args:
            storage_dir: Directory for storing records
        """
        self.storage_dir = storage_dir

    async def _ensure_dir(self) -> None:
        """Ensure storage directory exists."""
        await aiofiles.os.makedirs(self.storage_dir, exist_ok=True)

    def _get_record_path(self, key: str) -> Path:
        """Get path for a record."""
        # Use first 2 chars for distribution
        return self.storage_dir / key[:2] / f"{key}.json"

    def _get_run_index_path(self, run_id: str) -> Path:
        """Get path for run index."""
        return self.storage_dir / f"run-{run_id}.json"

    async def record(
        self,
        key: str,
        run_id: str,
        task_id: str,
        effect_type: EffectType,
        attempt: int,
        result_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IdempotencyRecord:
        """Record an executed operation.

        Args:
            key: Idempotency key
            run_id: Run identifier
            task_id: Task identifier
            effect_type: Type of effect
            attempt: Attempt number
            result_hash: Hash of result
            metadata: Additional metadata

        Returns:
            Created IdempotencyRecord
        """
        await self._ensure_dir()

        record = IdempotencyRecord(
            key=key,
            run_id=run_id,
            task_id=task_id,
            effect_type=effect_type,
            attempt=attempt,
            result_hash=result_hash,
            metadata=metadata or {},
        )

        # Store record
        record_path = self._get_record_path(key)
        await aiofiles.os.makedirs(record_path.parent, exist_ok=True)
        async with aiofiles.open(record_path, "w") as f:
            await f.write(json.dumps(record.to_dict(), indent=2))

        # Update run index
        await self._add_to_run_index(run_id, key)

        return record

    async def check(self, key: str) -> IdempotencyRecord | None:
        """Check if an operation was already executed.

        Args:
            key: Idempotency key to check

        Returns:
            IdempotencyRecord if found, None otherwise
        """
        record_path = self._get_record_path(key)
        if not record_path.exists():
            return None

        async with aiofiles.open(record_path, "r") as f:
            data = json.loads(await f.read())
        return IdempotencyRecord.from_dict(data)

    async def is_executed(self, key: str) -> bool:
        """Check if operation was already executed.

        Args:
            key: Idempotency key

        Returns:
            True if already executed
        """
        return await self.check(key) is not None

    async def list_run_records(self, run_id: str) -> list[IdempotencyRecord]:
        """List all idempotency records for a run.

        Args:
            run_id: Run to list records for

        Returns:
            List of IdempotencyRecords
        """
        index_path = self._get_run_index_path(run_id)
        if not index_path.exists():
            return []

        async with aiofiles.open(index_path, "r") as f:
            keys = json.loads(await f.read())

        records: list[IdempotencyRecord] = []
        for key in keys:
            record = await self.check(key)
            if record:
                records.append(record)

        return records

    async def clear_run(self, run_id: str) -> int:
        """Clear all records for a run.

        Args:
            run_id: Run to clear

        Returns:
            Number of records cleared
        """
        records = await self.list_run_records(run_id)
        count = 0

        for record in records:
            record_path = self._get_record_path(record.key)
            if record_path.exists():
                await aiofiles.os.remove(record_path)
                count += 1

        # Remove run index
        index_path = self._get_run_index_path(run_id)
        if index_path.exists():
            await aiofiles.os.remove(index_path)

        return count

    async def _add_to_run_index(self, run_id: str, key: str) -> None:
        """Add key to run index."""
        index_path = self._get_run_index_path(run_id)

        keys: list[str] = []
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                keys = json.loads(await f.read())

        if key not in keys:
            keys.append(key)
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(keys, indent=2))


T = TypeVar("T")


class IdempotentExecutor:
    """Executor that ensures idempotent operation execution.

    Wraps operations with idempotency checking and recording.
    """

    def __init__(self, store: IdempotencyStore) -> None:
        """Initialize the executor.

        Args:
            store: Idempotency store for tracking
        """
        self.store = store
        self.key_generator = IdempotencyKeyGenerator()

    async def execute_once(
        self,
        key: str,
        run_id: str,
        task_id: str,
        effect_type: EffectType,
        attempt: int,
        operation: Callable[[], T],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[T | None, bool]:
        """Execute an operation exactly once.

        If the operation was already executed (based on key),
        returns None and True. Otherwise executes and returns
        the result and False.

        Args:
            key: Idempotency key
            run_id: Run identifier
            task_id: Task identifier
            effect_type: Type of effect
            attempt: Attempt number
            operation: Operation to execute
            metadata: Additional metadata

        Returns:
            Tuple of (result or None, was_skipped)
        """
        # Check if already executed
        existing = await self.store.check(key)
        if existing:
            return None, True

        # Execute operation
        result = operation()

        # Record execution
        await self.store.record(
            key=key,
            run_id=run_id,
            task_id=task_id,
            effect_type=effect_type,
            attempt=attempt,
            metadata=metadata,
        )

        return result, False

    async def execute_once_async(
        self,
        key: str,
        run_id: str,
        task_id: str,
        effect_type: EffectType,
        attempt: int,
        operation: Callable[[], Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        """Execute an async operation exactly once.

        Args:
            key: Idempotency key
            run_id: Run identifier
            task_id: Task identifier
            effect_type: Type of effect
            attempt: Attempt number
            operation: Async operation to execute
            metadata: Additional metadata

        Returns:
            Tuple of (result or None, was_skipped)
        """
        # Check if already executed
        existing = await self.store.check(key)
        if existing:
            return None, True

        # Execute operation
        result = await operation()

        # Record execution
        await self.store.record(
            key=key,
            run_id=run_id,
            task_id=task_id,
            effect_type=effect_type,
            attempt=attempt,
            metadata=metadata,
        )

        return result, False
