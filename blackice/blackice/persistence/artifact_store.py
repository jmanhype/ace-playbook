"""Artifact storage for BLACKICE 3.0.

Provides durable storage for workspace artifacts:
- Code files, tests, documentation
- Build outputs and logs
- Intermediate work products
- Content-addressed storage with deduplication
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import aiofiles
import aiofiles.os


class ArtifactType(str, Enum):
    """Types of artifacts."""

    SOURCE = "source"  # Source code files
    TEST = "test"  # Test files
    DOC = "doc"  # Documentation
    CONFIG = "config"  # Configuration files
    LOG = "log"  # Execution logs
    OUTPUT = "output"  # Build outputs
    CHECKPOINT = "checkpoint"  # State checkpoints
    OTHER = "other"  # Other artifacts


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact.

    Attributes:
        id: Unique artifact identifier
        run_id: Run this artifact belongs to
        path: Original file path (relative to workspace)
        artifact_type: Type of artifact
        content_hash: SHA-256 hash of content
        size_bytes: Content size in bytes
        created_at: When artifact was stored
        encoding: Content encoding (utf-8, binary)
        mime_type: MIME type if known
        metadata: Additional metadata
    """

    id: str
    run_id: str
    path: str
    artifact_type: ArtifactType
    content_hash: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    encoding: str = "utf-8"
    mime_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "path": self.path,
            "artifact_type": self.artifact_type.value,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "encoding": self.encoding,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            path=data["path"],
            artifact_type=ArtifactType(data["artifact_type"]),
            content_hash=data["content_hash"],
            size_bytes=data["size_bytes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            encoding=data.get("encoding", "utf-8"),
            mime_type=data.get("mime_type", "application/octet-stream"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ArtifactStoreConfig:
    """Configuration for the artifact store."""

    storage_dir: Path = field(
        default_factory=lambda: Path.home() / ".blackice" / "artifacts"
    )
    content_dir: str = "content"  # Subdirectory for content-addressed storage
    metadata_dir: str = "metadata"  # Subdirectory for metadata
    deduplicate: bool = True  # Enable content deduplication


class ArtifactStore:
    """Content-addressed artifact storage.

    Stores artifacts with:
    - Content-addressed storage for deduplication
    - Metadata tracking for each artifact
    - Per-run organization
    - Hash verification on retrieval
    """

    def __init__(self, config: ArtifactStoreConfig | None = None) -> None:
        """Initialize the artifact store.

        Args:
            config: Store configuration
        """
        self.config = config or ArtifactStoreConfig()
        self._initialized = False

    @property
    def _content_dir(self) -> Path:
        """Get content storage directory."""
        return self.config.storage_dir / self.config.content_dir

    @property
    def _metadata_dir(self) -> Path:
        """Get metadata storage directory."""
        return self.config.storage_dir / self.config.metadata_dir

    async def initialize(self) -> None:
        """Initialize the artifact store."""
        if self._initialized:
            return

        await aiofiles.os.makedirs(self._content_dir, exist_ok=True)
        await aiofiles.os.makedirs(self._metadata_dir, exist_ok=True)
        self._initialized = True

    async def close(self) -> None:
        """Close the artifact store."""
        self._initialized = False

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _get_content_path(self, content_hash: str) -> Path:
        """Get storage path for content hash.

        Uses first 2 chars as subdirectory for better filesystem distribution.
        """
        return self._content_dir / content_hash[:2] / content_hash

    def _get_metadata_path(self, artifact_id: str) -> Path:
        """Get path for artifact metadata."""
        return self._metadata_dir / f"{artifact_id}.json"

    def _get_run_index_path(self, run_id: str) -> Path:
        """Get path for run artifact index."""
        return self._metadata_dir / f"run-{run_id}.json"

    async def store(
        self,
        run_id: str,
        path: str,
        content: bytes,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        encoding: str = "utf-8",
        mime_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        """Store an artifact.

        Args:
            run_id: Run this artifact belongs to
            path: Original file path
            content: File content
            artifact_type: Type of artifact
            encoding: Content encoding
            mime_type: MIME type
            metadata: Additional metadata

        Returns:
            ArtifactMetadata for the stored artifact
        """
        await self.initialize()

        # Compute content hash
        content_hash = self._compute_hash(content)

        # Store content (deduplicated)
        content_path = self._get_content_path(content_hash)
        if not content_path.exists():
            await aiofiles.os.makedirs(content_path.parent, exist_ok=True)
            async with aiofiles.open(content_path, "wb") as f:
                await f.write(content)

        # Create metadata
        artifact = ArtifactMetadata(
            id=str(uuid4()),
            run_id=run_id,
            path=path,
            artifact_type=artifact_type,
            content_hash=content_hash,
            size_bytes=len(content),
            encoding=encoding,
            mime_type=mime_type,
            metadata=metadata or {},
        )

        # Store metadata
        metadata_path = self._get_metadata_path(artifact.id)
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(json.dumps(artifact.to_dict(), indent=2))

        # Update run index
        await self._add_to_run_index(run_id, artifact.id)

        return artifact

    async def store_text(
        self,
        run_id: str,
        path: str,
        content: str,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        """Store a text artifact (convenience method).

        Args:
            run_id: Run this artifact belongs to
            path: Original file path
            content: Text content
            artifact_type: Type of artifact
            metadata: Additional metadata

        Returns:
            ArtifactMetadata for the stored artifact
        """
        return await self.store(
            run_id=run_id,
            path=path,
            content=content.encode("utf-8"),
            artifact_type=artifact_type,
            encoding="utf-8",
            mime_type="text/plain",
            metadata=metadata,
        )

    async def retrieve(self, artifact_id: str) -> tuple[ArtifactMetadata, bytes]:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: Artifact ID to retrieve

        Returns:
            Tuple of (metadata, content)

        Raises:
            FileNotFoundError: If artifact not found
            ValueError: If content hash doesn't match
        """
        # Load metadata
        metadata_path = self._get_metadata_path(artifact_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")

        async with aiofiles.open(metadata_path, "r") as f:
            data = json.loads(await f.read())
        artifact = ArtifactMetadata.from_dict(data)

        # Load content
        content_path = self._get_content_path(artifact.content_hash)
        if not content_path.exists():
            raise FileNotFoundError(
                f"Content not found for artifact: {artifact_id}"
            )

        async with aiofiles.open(content_path, "rb") as f:
            content = await f.read()

        # Verify hash
        actual_hash = self._compute_hash(content)
        if actual_hash != artifact.content_hash:
            raise ValueError(
                f"Content hash mismatch: expected {artifact.content_hash}, got {actual_hash}"
            )

        return artifact, content

    async def retrieve_text(self, artifact_id: str) -> tuple[ArtifactMetadata, str]:
        """Retrieve a text artifact (convenience method).

        Args:
            artifact_id: Artifact ID to retrieve

        Returns:
            Tuple of (metadata, text content)
        """
        artifact, content = await self.retrieve(artifact_id)
        return artifact, content.decode(artifact.encoding)

    async def get_metadata(self, artifact_id: str) -> ArtifactMetadata | None:
        """Get artifact metadata without loading content.

        Args:
            artifact_id: Artifact ID

        Returns:
            ArtifactMetadata or None if not found
        """
        metadata_path = self._get_metadata_path(artifact_id)
        if not metadata_path.exists():
            return None

        async with aiofiles.open(metadata_path, "r") as f:
            data = json.loads(await f.read())
        return ArtifactMetadata.from_dict(data)

    async def list_run_artifacts(
        self,
        run_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactMetadata]:
        """List all artifacts for a run.

        Args:
            run_id: Run to list artifacts for
            artifact_type: Optional filter by type

        Returns:
            List of ArtifactMetadata
        """
        index_path = self._get_run_index_path(run_id)
        if not index_path.exists():
            return []

        async with aiofiles.open(index_path, "r") as f:
            artifact_ids = json.loads(await f.read())

        artifacts: list[ArtifactMetadata] = []
        for artifact_id in artifact_ids:
            artifact = await self.get_metadata(artifact_id)
            if artifact:
                if artifact_type is None or artifact.artifact_type == artifact_type:
                    artifacts.append(artifact)

        return artifacts

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Note: Content is not deleted if shared with other artifacts.

        Args:
            artifact_id: Artifact ID to delete

        Returns:
            True if deleted, False if not found
        """
        metadata_path = self._get_metadata_path(artifact_id)
        if not metadata_path.exists():
            return False

        # Load metadata to get run_id
        async with aiofiles.open(metadata_path, "r") as f:
            data = json.loads(await f.read())
        artifact = ArtifactMetadata.from_dict(data)

        # Remove from run index
        await self._remove_from_run_index(artifact.run_id, artifact_id)

        # Delete metadata
        await aiofiles.os.remove(metadata_path)

        return True

    async def delete_run_artifacts(self, run_id: str) -> int:
        """Delete all artifacts for a run.

        Args:
            run_id: Run to delete artifacts for

        Returns:
            Number of artifacts deleted
        """
        artifacts = await self.list_run_artifacts(run_id)
        count = 0

        for artifact in artifacts:
            if await self.delete(artifact.id):
                count += 1

        # Delete run index
        index_path = self._get_run_index_path(run_id)
        if index_path.exists():
            await aiofiles.os.remove(index_path)

        return count

    async def _add_to_run_index(self, run_id: str, artifact_id: str) -> None:
        """Add artifact to run index."""
        index_path = self._get_run_index_path(run_id)

        artifact_ids: list[str] = []
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                artifact_ids = json.loads(await f.read())

        if artifact_id not in artifact_ids:
            artifact_ids.append(artifact_id)
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(artifact_ids, indent=2))

    async def _remove_from_run_index(self, run_id: str, artifact_id: str) -> None:
        """Remove artifact from run index."""
        index_path = self._get_run_index_path(run_id)
        if not index_path.exists():
            return

        async with aiofiles.open(index_path, "r") as f:
            artifact_ids = json.loads(await f.read())

        if artifact_id in artifact_ids:
            artifact_ids.remove(artifact_id)
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(artifact_ids, indent=2))

    async def compute_run_hashes(self, run_id: str) -> dict[str, str]:
        """Compute content hashes for all artifacts in a run.

        Useful for receipt generation.

        Args:
            run_id: Run to compute hashes for

        Returns:
            Dictionary of path -> content_hash
        """
        artifacts = await self.list_run_artifacts(run_id)
        return {artifact.path: artifact.content_hash for artifact in artifacts}

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        total_size = 0
        content_count = 0
        metadata_count = 0

        # Count content files
        if self._content_dir.exists():
            for subdir in self._content_dir.iterdir():
                if subdir.is_dir():
                    for content_file in subdir.iterdir():
                        content_count += 1
                        total_size += content_file.stat().st_size

        # Count metadata files
        if self._metadata_dir.exists():
            for meta_file in self._metadata_dir.glob("*.json"):
                if not meta_file.name.startswith("run-"):
                    metadata_count += 1

        return {
            "content_files": content_count,
            "metadata_files": metadata_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_dir": str(self.config.storage_dir),
        }
