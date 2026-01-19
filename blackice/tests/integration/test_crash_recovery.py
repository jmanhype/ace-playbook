"""Integration tests for crash recovery (IT-002).

Tests the crash recovery and resume functionality:
- Mid-run termination and resumption
- Completed tasks are skipped on resume
- In-flight tasks are retried with fresh attempt IDs
- Event store persistence across crashes
- Hash chain integrity verification
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.primitives.types import EventType, RunId, new_run_id, TaskId, new_task_id
from blackice.recovery.checkpoint import CheckpointManager
from blackice.recovery.resume import ResumeManager
from blackice.schemas.event import Event, EventLog, EventPayloads


class TestEventStorePersistence:
    """Test event store persistence across restarts."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> RunId:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.mark.asyncio
    async def test_events_persist_across_restarts(
        self, temp_dir: Path, run_id: RunId
    ) -> None:
        """Events written are readable after store restart."""
        config = EventStoreConfig(storage_dir=temp_dir)

        # First session: write some events
        store1 = EventStore(config)
        await store1.initialize()

        await store1.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started(
                vision="Test vision",
                edition="core",
                config={},
            ),
        )
        await store1.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("task-1", attempt=1),
        )
        await store1.close()

        # Second session: read events back
        store2 = EventStore(config)
        await store2.initialize()

        events = await store2.get_events(run_id)
        assert len(events) == 2
        assert events[0].type == EventType.RUN_STARTED
        assert events[1].type == EventType.TASK_STARTED

        await store2.close()

    @pytest.mark.asyncio
    async def test_hash_chain_integrity_verified_on_load(
        self, temp_dir: Path, run_id: RunId
    ) -> None:
        """Hash chain integrity is verified when loading events."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()

        # Write events
        await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )
        await store.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("task-1", 1),
        )
        await store.close()

        # Verify integrity
        store2 = EventStore(config)
        await store2.initialize()
        is_valid, error = await store2.verify_integrity(run_id)
        assert is_valid is True
        assert error is None
        await store2.close()

    @pytest.mark.asyncio
    async def test_events_ordered_by_sequence(
        self, temp_dir: Path, run_id: RunId
    ) -> None:
        """Events are retrieved in sequence order."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()

        # Write multiple events
        for i in range(5):
            await store.append(
                run_id,
                EventType.TASK_STARTED,
                EventPayloads.task_started(f"task-{i}", 1),
            )

        events = await store.get_events(run_id)
        assert len(events) == 5
        for i, event in enumerate(events):
            assert event.sequence == i

        await store.close()


class TestCheckpointCreation:
    """Test checkpoint creation for crash recovery points."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> RunId:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    def event_store(self, temp_dir: Path) -> EventStore:
        """Create an event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_checkpoint_captures_current_state(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Checkpoints capture current event sequence."""
        await event_store.initialize()
        checkpoint_mgr = CheckpointManager(event_store, temp_dir / "checkpoints")

        # Add some events
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )
        await event_store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("task-1", 1.5),
        )

        # Create checkpoint
        checkpoint = await checkpoint_mgr.create_checkpoint(run_id)

        assert checkpoint.run_id == run_id
        assert checkpoint.event_sequence == 1  # 0-indexed last event
        assert checkpoint.completed_tasks == ["task-1"]

        await event_store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_lists_completed_tasks(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Checkpoints list all completed tasks."""
        await event_store.initialize()
        checkpoint_mgr = CheckpointManager(event_store, temp_dir / "checkpoints")

        # Complete multiple tasks
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )

        for task_name in ["plan", "implement", "test"]:
            await event_store.append(
                run_id,
                EventType.TASK_STARTED,
                EventPayloads.task_started(task_name, 1),
            )
            await event_store.append(
                run_id,
                EventType.TASK_COMPLETED,
                EventPayloads.task_completed(task_name, 1.0),
            )

        checkpoint = await checkpoint_mgr.create_checkpoint(run_id)
        assert set(checkpoint.completed_tasks) == {"plan", "implement", "test"}

        await event_store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_stored_to_disk(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Checkpoints are persisted to disk."""
        await event_store.initialize()
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_mgr = CheckpointManager(event_store, checkpoint_dir)

        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )

        checkpoint = await checkpoint_mgr.create_checkpoint(run_id)

        # Verify file exists
        checkpoint_file = checkpoint_dir / str(run_id) / f"checkpoint-{checkpoint.id}.json"
        assert checkpoint_file.exists()

        await event_store.close()


class TestResumeFromCrash:
    """Test resuming from crash state."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> RunId:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    def event_store(self, temp_dir: Path) -> EventStore:
        """Create an event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_resume_skips_completed_tasks(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Completed tasks are skipped on resume."""
        await event_store.initialize()
        resume_mgr = ResumeManager(event_store, temp_dir / "checkpoints")

        # Simulate first run: complete some tasks
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )
        await event_store.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("plan", 1),
        )
        await event_store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("plan", 2.0),
        )

        # Get resume state
        resume_state = await resume_mgr.get_resume_state(run_id)

        assert "plan" in resume_state.completed_tasks
        assert resume_state.should_skip_task("plan") is True
        assert resume_state.should_skip_task("implement") is False

        await event_store.close()

    @pytest.mark.asyncio
    async def test_resume_retries_in_flight_tasks_with_new_attempt(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """In-flight tasks are retried with fresh attempt IDs."""
        await event_store.initialize()
        resume_mgr = ResumeManager(event_store, temp_dir / "checkpoints")

        # Simulate crash during task
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )
        await event_store.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("implement", 1),
        )
        # No TASK_COMPLETED - simulates crash

        resume_state = await resume_mgr.get_resume_state(run_id)

        assert resume_state.should_skip_task("implement") is False
        assert "implement" in resume_state.in_flight_tasks
        assert resume_state.get_next_attempt("implement") == 2

        await event_store.close()

    @pytest.mark.asyncio
    async def test_resume_identifies_last_checkpoint(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Resume finds the most recent checkpoint."""
        await event_store.initialize()
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_mgr = CheckpointManager(event_store, checkpoint_dir)
        resume_mgr = ResumeManager(event_store, checkpoint_dir)

        # Create run with checkpoint
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )
        await event_store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("phase-1", 1.0),
        )

        checkpoint1 = await checkpoint_mgr.create_checkpoint(run_id)

        await event_store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("phase-2", 1.0),
        )

        checkpoint2 = await checkpoint_mgr.create_checkpoint(run_id)

        # Resume should use latest checkpoint
        resume_state = await resume_mgr.get_resume_state(run_id)
        assert resume_state.last_checkpoint_id == checkpoint2.id

        await event_store.close()

    @pytest.mark.asyncio
    async def test_resume_without_checkpoint_starts_from_beginning(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Without checkpoints, resume replays from beginning."""
        await event_store.initialize()
        resume_mgr = ResumeManager(event_store, temp_dir / "checkpoints")

        # Just start events, no checkpoint
        await event_store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "core", {}),
        )

        resume_state = await resume_mgr.get_resume_state(run_id)

        assert resume_state.last_checkpoint_id is None
        assert resume_state.resume_from_sequence == 0

        await event_store.close()


class TestIdempotencyKeys:
    """Test idempotency key handling for external effects."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> RunId:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    def event_store(self, temp_dir: Path) -> EventStore:
        """Create an event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_idempotency_key_prevents_duplicate_execution(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Idempotency keys prevent duplicate side effects."""
        await event_store.initialize()
        resume_mgr = ResumeManager(event_store, temp_dir / "checkpoints")

        # Record a command execution with idempotency key
        idem_key = "write-config-file-v1"
        await event_store.append(
            run_id,
            EventType.COMMAND_EXECUTED,
            {
                **EventPayloads.command_executed("echo test", 0, 0.1),
                "idempotency_key": idem_key,
            },
        )

        # Check if command should be skipped on resume
        resume_state = await resume_mgr.get_resume_state(run_id)

        assert resume_state.is_idempotent_key_used(idem_key) is True
        assert resume_state.is_idempotent_key_used("different-key") is False

        await event_store.close()

    @pytest.mark.asyncio
    async def test_different_attempts_have_unique_keys(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Different task attempts generate unique idempotency keys."""
        await event_store.initialize()
        resume_mgr = ResumeManager(event_store, temp_dir / "checkpoints")

        # Generate keys for different attempts
        task_id = new_task_id()
        key1 = resume_mgr.generate_idempotency_key(run_id, task_id, attempt=1)
        key2 = resume_mgr.generate_idempotency_key(run_id, task_id, attempt=2)

        assert key1 != key2
        assert str(task_id) in key1
        assert str(task_id) in key2

        await event_store.close()


class TestDeadLetterHandling:
    """Test dead letter queue for unprocessable events."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> RunId:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    def event_store(self, temp_dir: Path) -> EventStore:
        """Create an event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_failed_tasks_after_max_retries_go_to_dead_letter(
        self, temp_dir: Path, run_id: RunId, event_store: EventStore
    ) -> None:
        """Tasks that fail max retries are moved to dead letter queue."""
        from blackice.recovery.dead_letter import DeadLetterQueue

        await event_store.initialize()
        dead_letter = DeadLetterQueue(temp_dir / "dead-letter")

        # Simulate max retry failures
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            await event_store.append(
                run_id,
                EventType.TASK_STARTED,
                EventPayloads.task_started("failing-task", attempt),
            )
            await event_store.append(
                run_id,
                EventType.TASK_FAILED,
                EventPayloads.task_failed("failing-task", "Connection timeout", attempt),
            )

        # Add to dead letter queue
        await dead_letter.add(
            run_id=run_id,
            task_name="failing-task",
            reason="Max retries exceeded",
            last_error="Connection timeout",
            attempts=max_attempts,
        )

        # Verify in dead letter queue
        entries = await dead_letter.list_entries(run_id)
        assert len(entries) == 1
        assert entries[0].task_name == "failing-task"
        assert entries[0].attempts == max_attempts

        await event_store.close()

    @pytest.mark.asyncio
    async def test_dead_letter_entries_can_be_retried_manually(
        self, temp_dir: Path, run_id: RunId
    ) -> None:
        """Dead letter entries can be marked for retry."""
        from blackice.recovery.dead_letter import DeadLetterQueue

        dead_letter = DeadLetterQueue(temp_dir / "dead-letter")

        await dead_letter.add(
            run_id=run_id,
            task_name="failing-task",
            reason="Test failure",
            last_error="Some error",
            attempts=3,
        )

        entries = await dead_letter.list_entries(run_id)
        entry_id = entries[0].id

        # Mark for retry
        await dead_letter.mark_for_retry(entry_id)

        updated_entries = await dead_letter.list_entries(run_id)
        assert updated_entries[0].retry_requested is True

        # Remove from queue after processing
        await dead_letter.remove(entry_id)
        remaining = await dead_letter.list_entries(run_id)
        assert len(remaining) == 0
