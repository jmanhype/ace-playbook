"""Persistence layer for BLACKICE 3.0.

Provides durable storage for:
- Events (append-only log with hash chain)
- Artifacts (workspace files and outputs)
- Schema versioning and migrations
- State projections for reconstruction
"""

from blackice.persistence.artifact_store import (
    ArtifactMetadata,
    ArtifactStore,
    ArtifactStoreConfig,
    ArtifactType,
)
from blackice.persistence.event_schema import (
    EventMigrator,
    Migration,
    MigrationRegistry,
    SchemaInfo,
    SchemaValidator,
    SchemaVersion,
    add_schema_version,
)
from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.persistence.projections import (
    RunProjection,
    RunState,
    RunStatus,
    TaskProjection,
    TaskState,
    TaskStatus,
    reconstruct_run_state,
    reconstruct_task_states,
)

__all__ = [
    # Event Store
    "EventStore",
    "EventStoreConfig",
    # Schema Versioning
    "SchemaVersion",
    "Migration",
    "MigrationRegistry",
    "SchemaValidator",
    "SchemaInfo",
    "EventMigrator",
    "add_schema_version",
    # Projections
    "RunProjection",
    "RunState",
    "RunStatus",
    "TaskProjection",
    "TaskState",
    "TaskStatus",
    "reconstruct_run_state",
    "reconstruct_task_states",
    # Artifact Store
    "ArtifactStore",
    "ArtifactStoreConfig",
    "ArtifactMetadata",
    "ArtifactType",
]
