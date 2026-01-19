"""Unit tests for EventStore.

Tests the event store functionality:
- JSONL persistence format
- Hash chain creation and verification
- Event sequencing
- Event retrieval and filtering
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.primitives.types import EventType, new_run_id
from blackice.schemas.event import Event, EventPayloads


class TestEventStoreConfig:
    """Test EventStoreConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Default config has sensible defaults."""
        config = EventStoreConfig()
        assert config.storage_dir is not None
        assert config.file_extension == ".jsonl"
        assert config.sync_on_write is True

    def test_custom_storage_dir(self, tmp_path: Path) -> None:
        """Custom storage directory is respected."""
        config = EventStoreConfig(storage_dir=tmp_path / "events")
        assert config.storage_dir == tmp_path / "events"

    def test_sync_disabled(self) -> None:
        """Sync can be disabled for performance."""
        config = EventStoreConfig(sync_on_write=False)
        assert config.sync_on_write is False


class TestEventStoreInitialization:
    """Test EventStore initialization."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.mark.asyncio
    async def test_creates_storage_directory(self, temp_dir: Path) -> None:
        """Storage directory is created if it doesn't exist."""
        storage_dir = temp_dir / "events"
        assert not storage_dir.exists()

        config = EventStoreConfig(storage_dir=storage_dir)
        store = EventStore(config)
        await store.initialize()

        assert storage_dir.exists()
        await store.close()

    @pytest.mark.asyncio
    async def test_handles_existing_directory(self, temp_dir: Path) -> None:
        """Existing directory is handled gracefully."""
        storage_dir = temp_dir / "events"
        storage_dir.mkdir(parents=True)

        config = EventStoreConfig(storage_dir=storage_dir)
        store = EventStore(config)
        await store.initialize()

        assert storage_dir.exists()
        await store.close()


class TestEventAppend:
    """Test appending events to the store."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> str:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_append_creates_event_with_id(self, store: EventStore, run_id: str) -> None:
        """Appended events have unique IDs."""
        event = await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )

        assert event.id is not None
        # Event ID is a UUID, check it has content
        assert str(event.id) != ""

    @pytest.mark.asyncio
    async def test_append_increments_sequence(self, store: EventStore, run_id: str) -> None:
        """Event sequences increment correctly."""
        e1 = await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )
        e2 = await store.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("task-1", 1),
        )
        e3 = await store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("task-1", 1.5),
        )

        assert e1.sequence == 0
        assert e2.sequence == 1
        assert e3.sequence == 2

    @pytest.mark.asyncio
    async def test_append_links_hash_chain(self, store: EventStore, run_id: str) -> None:
        """Events are linked via hash chain."""
        e1 = await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )
        e2 = await store.append(
            run_id,
            EventType.TASK_STARTED,
            EventPayloads.task_started("task-1", 1),
        )

        assert e1.previous_hash is None
        assert e2.previous_hash == e1.hash

    @pytest.mark.asyncio
    async def test_append_sets_run_id(self, store: EventStore, run_id: str) -> None:
        """Events are associated with the correct run."""
        event = await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )

        assert event.run_id == run_id

    @pytest.mark.asyncio
    async def test_append_stores_payload(self, store: EventStore, run_id: str) -> None:
        """Event payload is stored correctly."""
        payload = EventPayloads.run_started("My Vision", "enterprise", {"key": "value"})
        event = await store.append(run_id, EventType.RUN_STARTED, payload)

        assert event.payload["vision_preview"] == "My Vision"
        assert event.payload["edition"] == "enterprise"


class TestEventRetrieval:
    """Test retrieving events from the store."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> str:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_get_events_returns_all_for_run(
        self, store: EventStore, run_id: str
    ) -> None:
        """Get events returns all events for a run."""
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

        events = await store.get_events(run_id)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_empty_for_unknown_run(self, store: EventStore) -> None:
        """Get events returns empty for unknown run."""
        events = await store.get_events("unknown-run-id")
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_get_events_filters_by_type(
        self, store: EventStore, run_id: str
    ) -> None:
        """Get events can filter by event type."""
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
        await store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("task-1", 1.5),
        )

        task_events = await store.get_events(run_id, event_type=EventType.TASK_STARTED)
        assert len(task_events) == 1
        assert task_events[0].type == EventType.TASK_STARTED

    @pytest.mark.asyncio
    async def test_get_events_since_sequence(
        self, store: EventStore, run_id: str
    ) -> None:
        """Get events can start from a specific sequence."""
        for i in range(5):
            await store.append(
                run_id,
                EventType.TASK_STARTED,
                EventPayloads.task_started(f"task-{i}", 1),
            )

        events = await store.get_events(run_id, since_sequence=2)
        assert len(events) == 3
        assert events[0].sequence == 2

    @pytest.mark.asyncio
    async def test_get_latest_event(self, store: EventStore, run_id: str) -> None:
        """Get latest event returns most recent."""
        await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )
        await store.append(
            run_id,
            EventType.TASK_COMPLETED,
            EventPayloads.task_completed("task-1", 1.5),
        )

        latest = await store.get_latest_event(run_id)
        assert latest is not None
        assert latest.type == EventType.TASK_COMPLETED


class TestHashChainVerification:
    """Test hash chain integrity verification."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> str:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_verify_valid_chain(self, store: EventStore, run_id: str) -> None:
        """Valid hash chain passes verification."""
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

        is_valid, error = await store.verify_integrity(run_id)
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_empty_chain(self, store: EventStore) -> None:
        """Empty chain is valid."""
        is_valid, error = await store.verify_integrity("empty-run")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_single_event_chain(
        self, store: EventStore, run_id: str
    ) -> None:
        """Single event chain is valid."""
        await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test", "lite", {}),
        )

        is_valid, error = await store.verify_integrity(run_id)
        assert is_valid is True


class TestJSONLFormat:
    """Test JSONL file format."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    def run_id(self) -> str:
        """Create a unique run ID."""
        return new_run_id()

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_events_written_as_jsonl(
        self, temp_dir: Path, run_id: str, store: EventStore
    ) -> None:
        """Events are written as JSONL (one JSON object per line)."""
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

        # Read the raw file
        event_file = temp_dir / f"{run_id}.jsonl"
        lines = event_file.read_text().strip().split("\n")

        assert len(lines) == 2
        for line in lines:
            # Each line should be valid JSON
            obj = json.loads(line)
            assert "id" in obj
            assert "type" in obj
            assert "sequence" in obj

    @pytest.mark.asyncio
    async def test_jsonl_includes_all_event_fields(
        self, temp_dir: Path, run_id: str, store: EventStore
    ) -> None:
        """JSONL includes all required event fields."""
        await store.append(
            run_id,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Test Vision", "enterprise", {"config": "value"}),
        )

        event_file = temp_dir / f"{run_id}.jsonl"
        line = event_file.read_text().strip()
        obj = json.loads(line)

        # run_id is stored as string in JSONL
        assert obj["run_id"] == str(run_id)
        assert obj["type"] == EventType.RUN_STARTED.value
        assert obj["payload"]["vision_preview"] == "Test Vision"
        assert obj["sequence"] == 0
        assert "hash" in obj or "previous_hash" in obj  # Hash chain fields


class TestMultiRunIsolation:
    """Test isolation between different runs."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_runs_have_separate_event_files(
        self, temp_dir: Path, store: EventStore
    ) -> None:
        """Different runs have separate event files."""
        run1 = new_run_id()
        run2 = new_run_id()

        await store.append(
            run1,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 1", "lite", {}),
        )
        await store.append(
            run2,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 2", "lite", {}),
        )

        file1 = temp_dir / f"{run1}.jsonl"
        file2 = temp_dir / f"{run2}.jsonl"

        assert file1.exists()
        assert file2.exists()
        assert file1 != file2

    @pytest.mark.asyncio
    async def test_run_sequences_independent(self, store: EventStore) -> None:
        """Event sequences are independent per run."""
        run1 = new_run_id()
        run2 = new_run_id()

        e1 = await store.append(
            run1,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 1", "lite", {}),
        )
        e2 = await store.append(
            run2,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 2", "lite", {}),
        )

        assert e1.sequence == 0
        assert e2.sequence == 0  # Each run starts at 0

    @pytest.mark.asyncio
    async def test_get_events_isolates_runs(self, store: EventStore) -> None:
        """Get events only returns events for specified run."""
        run1 = new_run_id()
        run2 = new_run_id()

        await store.append(
            run1,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 1", "lite", {}),
        )
        await store.append(
            run1,
            EventType.TASK_STARTED,
            EventPayloads.task_started("task-1", 1),
        )
        await store.append(
            run2,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 2", "lite", {}),
        )

        events1 = await store.get_events(run1)
        events2 = await store.get_events(run2)

        assert len(events1) == 2
        assert len(events2) == 1
        assert all(e.run_id == run1 for e in events1)
        assert all(e.run_id == run2 for e in events2)


class TestEventStoreListRuns:
    """Test listing available runs."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture
    async def store(self, temp_dir: Path) -> EventStore:
        """Create an initialized event store."""
        config = EventStoreConfig(storage_dir=temp_dir)
        store = EventStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_list_runs_returns_all(self, store: EventStore) -> None:
        """List runs returns all runs with events."""
        run1 = new_run_id()
        run2 = new_run_id()
        run3 = new_run_id()

        await store.append(
            run1,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 1", "lite", {}),
        )
        await store.append(
            run2,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 2", "core", {}),
        )
        await store.append(
            run3,
            EventType.RUN_STARTED,
            EventPayloads.run_started("Run 3", "enterprise", {}),
        )

        runs = await store.list_runs()
        assert len(runs) == 3
        # list_runs returns string IDs
        assert str(run1) in runs
        assert str(run2) in runs
        assert str(run3) in runs

    @pytest.mark.asyncio
    async def test_list_runs_empty_store(self, store: EventStore) -> None:
        """List runs returns empty for new store."""
        runs = await store.list_runs()
        assert len(runs) == 0
