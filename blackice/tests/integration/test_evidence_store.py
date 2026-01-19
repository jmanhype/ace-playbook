"""Integration tests for EvidenceStore (IT-007a).

Tests evidence artifact persistence including:
- Store initialization and cleanup
- Evidence storage with hash computation
- Retrieval by ID, run, and type
- Evidence collections
- Integrity verification
- Deletion with index cleanup
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from blackice.persistence import (
    AsyncEvidenceStore,
    EvidenceStore,
    EvidenceStoreConfig,
)
from blackice.primitives.types import RunId
from blackice.schemas.evidence import (
    CommandOutput,
    Evidence,
    EvidenceType,
    LintReport,
    SecurityScan,
    TestReport,
)


def make_run_id() -> RunId:
    """Create a valid RunId (UUID)."""
    return RunId(uuid4())


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "evidence"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def evidence_store(temp_storage_dir: Path) -> EvidenceStore:
    """Create an evidence store with temporary storage."""
    config = EvidenceStoreConfig(storage_dir=temp_storage_dir)
    return EvidenceStore(config)


class TestEvidenceStoreInitialization:
    """Test EvidenceStore initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_directories(
        self, temp_storage_dir: Path
    ) -> None:
        """Initialize should create storage directories."""
        config = EvidenceStoreConfig(storage_dir=temp_storage_dir)
        store = EvidenceStore(config)

        await store.initialize()

        assert (temp_storage_dir / "items").exists()
        assert (temp_storage_dir / "index").exists()

    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(self, evidence_store: EvidenceStore) -> None:
        """Initialize should be idempotent."""
        await evidence_store.initialize()
        await evidence_store.initialize()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_resets_state(self, evidence_store: EvidenceStore) -> None:
        """Close should reset initialized state."""
        await evidence_store.initialize()
        await evidence_store.close()
        assert evidence_store._initialized is False


class TestEvidenceStorage:
    """Test evidence storage operations."""

    @pytest.mark.asyncio
    async def test_store_evidence(self, evidence_store: EvidenceStore) -> None:
        """Should store evidence with computed hash."""
        run_id = make_run_id()
        report = TestReport(
            framework="pytest",
            total_tests=10,
            passed=10,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=5.0,
        )
        evidence = Evidence.from_test_report("ev-001", run_id, report)

        stored = await evidence_store.store(evidence)

        assert stored.id == "ev-001"
        assert stored.content_hash is not None

    @pytest.mark.asyncio
    async def test_get_stored_evidence(self, evidence_store: EvidenceStore) -> None:
        """Should retrieve stored evidence by ID."""
        run_id = make_run_id()
        report = TestReport(
            framework="pytest",
            total_tests=5,
            passed=5,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=1.0,
        )
        evidence = Evidence.from_test_report("ev-002", run_id, report)
        await evidence_store.store(evidence)

        retrieved = await evidence_store.get("ev-002")

        assert retrieved is not None
        assert retrieved.id == "ev-002"
        assert retrieved.test_report is not None
        assert retrieved.test_report.total_tests == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_evidence(
        self, evidence_store: EvidenceStore
    ) -> None:
        """Should return None for nonexistent evidence."""
        await evidence_store.initialize()
        retrieved = await evidence_store.get("nonexistent")
        assert retrieved is None


class TestEvidenceIndexing:
    """Test evidence indexing by run and type."""

    @pytest.mark.asyncio
    async def test_get_by_run(self, evidence_store: EvidenceStore) -> None:
        """Should retrieve all evidence for a run."""
        run_id = make_run_id()

        # Store multiple evidence items for same run
        evidence1 = Evidence.from_test_report(
            "ev-run-1",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=10,
                passed=10,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=5.0,
            ),
        )
        evidence2 = Evidence.from_lint_report(
            "ev-run-2",
            run_id,
            LintReport(
                linter="ruff",
                total_issues=0,
                errors=0,
                files_checked=20,
            ),
        )

        await evidence_store.store(evidence1)
        await evidence_store.store(evidence2)

        results = await evidence_store.get_by_run(run_id)

        assert len(results) == 2
        ids = {e.id for e in results}
        assert ids == {"ev-run-1", "ev-run-2"}

    @pytest.mark.asyncio
    async def test_get_by_run_empty(self, evidence_store: EvidenceStore) -> None:
        """Should return empty list for run with no evidence."""
        await evidence_store.initialize()
        results = await evidence_store.get_by_run(make_run_id())
        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_type(self, evidence_store: EvidenceStore) -> None:
        """Should retrieve all evidence of a type."""
        run_id1 = make_run_id()
        run_id2 = make_run_id()

        # Store test reports across runs
        evidence1 = Evidence.from_test_report(
            "ev-type-1",
            run_id1,
            TestReport(
                framework="pytest",
                total_tests=10,
                passed=10,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=5.0,
            ),
        )
        evidence2 = Evidence.from_test_report(
            "ev-type-2",
            run_id2,
            TestReport(
                framework="jest",
                total_tests=20,
                passed=18,
                failed=2,
                skipped=0,
                errors=0,
                duration_seconds=10.0,
            ),
        )
        evidence3 = Evidence.from_lint_report(
            "ev-type-3",
            run_id1,
            LintReport(
                linter="ruff",
                total_issues=5,
                errors=1,
                files_checked=10,
            ),
        )

        await evidence_store.store(evidence1)
        await evidence_store.store(evidence2)
        await evidence_store.store(evidence3)

        test_reports = await evidence_store.get_by_type(EvidenceType.TEST_REPORT)
        lint_reports = await evidence_store.get_by_type(EvidenceType.LINT_REPORT)

        assert len(test_reports) == 2
        assert len(lint_reports) == 1


class TestEvidenceCollection:
    """Test evidence collection retrieval."""

    @pytest.mark.asyncio
    async def test_get_collection(self, evidence_store: EvidenceStore) -> None:
        """Should retrieve evidence collection for a run."""
        run_id = make_run_id()

        evidence1 = Evidence.from_test_report(
            "ev-coll-1",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=10,
                passed=10,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=5.0,
            ),
        )
        evidence2 = Evidence.from_security_scan(
            "ev-coll-2",
            run_id,
            SecurityScan(
                scanner="bandit",
                total_findings=0,
                scan_duration_seconds=2.0,
            ),
        )

        await evidence_store.store(evidence1)
        await evidence_store.store(evidence2)

        collection = await evidence_store.get_collection(run_id)

        assert collection.run_id == run_id
        assert len(collection.evidence_items) == 2
        assert collection.all_passed is True


class TestEvidenceDeletion:
    """Test evidence deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_evidence(self, evidence_store: EvidenceStore) -> None:
        """Should delete evidence and update indexes."""
        run_id = make_run_id()
        evidence = Evidence.from_test_report(
            "ev-del-1",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=5,
                passed=5,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=1.0,
            ),
        )
        await evidence_store.store(evidence)

        deleted = await evidence_store.delete("ev-del-1")

        assert deleted is True
        assert await evidence_store.get("ev-del-1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, evidence_store: EvidenceStore) -> None:
        """Should return False for nonexistent evidence."""
        await evidence_store.initialize()
        deleted = await evidence_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_by_run(self, evidence_store: EvidenceStore) -> None:
        """Should delete all evidence for a run."""
        run_id = make_run_id()

        evidence1 = Evidence.from_test_report(
            "ev-delrun-1",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=5,
                passed=5,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=1.0,
            ),
        )
        evidence2 = Evidence.from_lint_report(
            "ev-delrun-2",
            run_id,
            LintReport(
                linter="ruff",
                total_issues=0,
                errors=0,
                files_checked=5,
            ),
        )

        await evidence_store.store(evidence1)
        await evidence_store.store(evidence2)

        deleted = await evidence_store.delete_by_run(run_id)

        assert deleted == 2
        assert await evidence_store.get_by_run(run_id) == []


class TestIntegrityVerification:
    """Test evidence integrity verification."""

    @pytest.mark.asyncio
    async def test_verify_integrity_valid(self, evidence_store: EvidenceStore) -> None:
        """Should verify intact evidence."""
        run_id = make_run_id()
        evidence = Evidence.from_test_report(
            "ev-int-1",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=5,
                passed=5,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=1.0,
            ),
        )
        await evidence_store.store(evidence)

        is_valid, message = await evidence_store.verify_integrity("ev-int-1")

        assert is_valid is True
        assert "verified" in message.lower()

    @pytest.mark.asyncio
    async def test_verify_integrity_nonexistent(
        self, evidence_store: EvidenceStore
    ) -> None:
        """Should fail for nonexistent evidence."""
        await evidence_store.initialize()
        is_valid, message = await evidence_store.verify_integrity("nonexistent")

        assert is_valid is False
        assert "not found" in message.lower()


class TestListRuns:
    """Test run listing operations."""

    @pytest.mark.asyncio
    async def test_list_runs(self, evidence_store: EvidenceStore) -> None:
        """Should list all runs with evidence."""
        run_id1 = make_run_id()
        run_id2 = make_run_id()

        evidence1 = Evidence.from_test_report(
            "ev-list-1",
            run_id1,
            TestReport(
                framework="pytest",
                total_tests=5,
                passed=5,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=1.0,
            ),
        )
        evidence2 = Evidence.from_test_report(
            "ev-list-2",
            run_id2,
            TestReport(
                framework="jest",
                total_tests=10,
                passed=10,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=2.0,
            ),
        )

        await evidence_store.store(evidence1)
        await evidence_store.store(evidence2)

        runs = await evidence_store.list_runs()

        assert len(runs) == 2
        assert str(run_id1) in runs
        assert str(run_id2) in runs

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, evidence_store: EvidenceStore) -> None:
        """Should return empty list when no evidence stored."""
        await evidence_store.initialize()
        runs = await evidence_store.list_runs()
        assert runs == []


class TestAsyncContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_storage_dir: Path) -> None:
        """Should work as async context manager."""
        config = EvidenceStoreConfig(storage_dir=temp_storage_dir)

        async with AsyncEvidenceStore(config) as store:
            run_id = make_run_id()
            evidence = Evidence.from_test_report(
                "ev-ctx-1",
                run_id,
                TestReport(
                    framework="pytest",
                    total_tests=1,
                    passed=1,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=0.1,
                ),
            )
            await store.store(evidence)
            retrieved = await store.get("ev-ctx-1")
            assert retrieved is not None
