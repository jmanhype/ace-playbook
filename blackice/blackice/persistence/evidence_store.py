"""Evidence storage for BLACKICE 3.0 Enterprise.

Provides durable storage for evidence artifacts:
- Test reports, security scans, lint reports
- Command outputs and build logs
- Content-addressed storage with deduplication
- Run-indexed retrieval
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import aiofiles
import aiofiles.os

from blackice.primitives.types import RunId
from blackice.schemas.evidence import (
    Evidence,
    EvidenceCollection,
    EvidenceStatus,
    EvidenceType,
)


@dataclass
class EvidenceStoreConfig:
    """Configuration for the evidence store."""

    storage_dir: Path = field(
        default_factory=lambda: Path.home() / ".blackice" / "evidence"
    )
    evidence_dir: str = "items"  # Subdirectory for evidence items
    index_dir: str = "index"  # Subdirectory for indexes
    compress: bool = False  # Compress evidence (future feature)


class EvidenceStore:
    """Storage for evidence artifacts.

    Stores evidence with:
    - JSON serialization of evidence objects
    - Run-indexed organization
    - Type-indexed retrieval
    - Hash verification
    """

    def __init__(self, config: EvidenceStoreConfig | None = None) -> None:
        """Initialize the evidence store.

        Args:
            config: Store configuration
        """
        self.config = config or EvidenceStoreConfig()
        self._initialized = False

    @property
    def _evidence_dir(self) -> Path:
        """Get evidence storage directory."""
        return self.config.storage_dir / self.config.evidence_dir

    @property
    def _index_dir(self) -> Path:
        """Get index storage directory."""
        return self.config.storage_dir / self.config.index_dir

    async def initialize(self) -> None:
        """Initialize the evidence store."""
        if self._initialized:
            return

        await aiofiles.os.makedirs(self._evidence_dir, exist_ok=True)
        await aiofiles.os.makedirs(self._index_dir, exist_ok=True)
        self._initialized = True

    async def close(self) -> None:
        """Close the evidence store."""
        self._initialized = False

    def _get_evidence_path(self, evidence_id: str) -> Path:
        """Get storage path for evidence.

        Uses first 2 chars of ID for filesystem distribution.
        """
        return self._evidence_dir / evidence_id[:2] / f"{evidence_id}.json"

    def _get_run_index_path(self, run_id: str) -> Path:
        """Get path for run evidence index."""
        return self._index_dir / f"run-{run_id}.json"

    def _get_type_index_path(self, evidence_type: EvidenceType) -> Path:
        """Get path for type-based index."""
        return self._index_dir / f"type-{evidence_type.value}.json"

    async def store(self, evidence: Evidence) -> Evidence:
        """Store an evidence item.

        Args:
            evidence: Evidence to store

        Returns:
            Evidence with computed hash
        """
        await self.initialize()

        # Compute hash if not set
        if evidence.content_hash is None:
            evidence.content_hash = evidence.compute_hash()

        # Serialize evidence
        evidence_data = evidence.model_dump(mode="json")

        # Store evidence file
        evidence_path = self._get_evidence_path(evidence.id)
        await aiofiles.os.makedirs(evidence_path.parent, exist_ok=True)
        async with aiofiles.open(evidence_path, "w") as f:
            await f.write(json.dumps(evidence_data, indent=2, default=str))

        # Update run index
        await self._add_to_index(
            self._get_run_index_path(str(evidence.run_id)),
            evidence.id,
        )

        # Update type index
        await self._add_to_index(
            self._get_type_index_path(evidence.evidence_type),
            evidence.id,
        )

        return evidence

    async def get(self, evidence_id: str) -> Evidence | None:
        """Retrieve an evidence item by ID.

        Args:
            evidence_id: Evidence ID

        Returns:
            Evidence if found, None otherwise
        """
        await self.initialize()

        evidence_path = self._get_evidence_path(evidence_id)
        if not evidence_path.exists():
            return None

        async with aiofiles.open(evidence_path, "r") as f:
            data = json.loads(await f.read())

        return Evidence.model_validate(data)

    async def get_by_run(self, run_id: RunId) -> list[Evidence]:
        """Get all evidence for a run.

        Args:
            run_id: Run ID

        Returns:
            List of evidence items
        """
        await self.initialize()

        index_path = self._get_run_index_path(str(run_id))
        if not index_path.exists():
            return []

        async with aiofiles.open(index_path, "r") as f:
            evidence_ids = json.loads(await f.read())

        evidence_items = []
        for eid in evidence_ids:
            evidence = await self.get(eid)
            if evidence:
                evidence_items.append(evidence)

        return evidence_items

    async def get_by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        """Get all evidence of a specific type.

        Args:
            evidence_type: Type of evidence

        Returns:
            List of evidence items
        """
        await self.initialize()

        index_path = self._get_type_index_path(evidence_type)
        if not index_path.exists():
            return []

        async with aiofiles.open(index_path, "r") as f:
            evidence_ids = json.loads(await f.read())

        evidence_items = []
        for eid in evidence_ids:
            evidence = await self.get(eid)
            if evidence:
                evidence_items.append(evidence)

        return evidence_items

    async def get_collection(self, run_id: RunId) -> EvidenceCollection:
        """Get an evidence collection for a run.

        Args:
            run_id: Run ID

        Returns:
            EvidenceCollection containing all evidence for the run
        """
        evidence_items = await self.get_by_run(run_id)
        collection = EvidenceCollection(run_id=run_id)
        for item in evidence_items:
            collection.add(item)
        return collection

    async def delete(self, evidence_id: str) -> bool:
        """Delete an evidence item.

        Args:
            evidence_id: Evidence ID

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        evidence_path = self._get_evidence_path(evidence_id)
        if not evidence_path.exists():
            return False

        # Load evidence to get run_id and type for index cleanup
        async with aiofiles.open(evidence_path, "r") as f:
            data = json.loads(await f.read())

        # Remove from indexes
        await self._remove_from_index(
            self._get_run_index_path(data["run_id"]),
            evidence_id,
        )
        await self._remove_from_index(
            self._get_type_index_path(EvidenceType(data["evidence_type"])),
            evidence_id,
        )

        # Delete evidence file
        await aiofiles.os.remove(evidence_path)

        return True

    async def delete_by_run(self, run_id: RunId) -> int:
        """Delete all evidence for a run.

        Args:
            run_id: Run ID

        Returns:
            Number of evidence items deleted
        """
        evidence_items = await self.get_by_run(run_id)
        deleted = 0
        for evidence in evidence_items:
            if await self.delete(evidence.id):
                deleted += 1
        return deleted

    async def list_runs(self) -> list[str]:
        """List all runs with stored evidence.

        Returns:
            List of run IDs
        """
        await self.initialize()

        runs = []
        if not self._index_dir.exists():
            return runs

        for file in self._index_dir.iterdir():
            if file.name.startswith("run-") and file.suffix == ".json":
                run_id = file.stem.replace("run-", "")
                runs.append(run_id)

        return sorted(runs)

    async def _add_to_index(self, index_path: Path, evidence_id: str) -> None:
        """Add evidence ID to an index file."""
        evidence_ids = []
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                evidence_ids = json.loads(await f.read())

        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(evidence_ids, indent=2))

    async def _remove_from_index(self, index_path: Path, evidence_id: str) -> None:
        """Remove evidence ID from an index file."""
        if not index_path.exists():
            return

        async with aiofiles.open(index_path, "r") as f:
            evidence_ids = json.loads(await f.read())

        if evidence_id in evidence_ids:
            evidence_ids.remove(evidence_id)
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(evidence_ids, indent=2))

    async def verify_integrity(self, evidence_id: str) -> tuple[bool, str]:
        """Verify integrity of stored evidence.

        Args:
            evidence_id: Evidence ID

        Returns:
            (is_valid, message) tuple
        """
        evidence = await self.get(evidence_id)
        if not evidence:
            return False, f"Evidence not found: {evidence_id}"

        stored_hash = evidence.content_hash
        if stored_hash is None:
            return False, "Evidence has no content hash"

        computed_hash = evidence.compute_hash()
        if stored_hash.value != computed_hash.value:
            return False, f"Hash mismatch: stored={stored_hash.value[:8]}... computed={computed_hash.value[:8]}..."

        return True, "Evidence integrity verified"


# Convenience async context manager
class AsyncEvidenceStore:
    """Async context manager for EvidenceStore."""

    def __init__(self, config: EvidenceStoreConfig | None = None) -> None:
        self.store = EvidenceStore(config)

    async def __aenter__(self) -> EvidenceStore:
        await self.store.initialize()
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.store.close()
