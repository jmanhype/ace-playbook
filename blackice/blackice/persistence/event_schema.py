"""Event schema versioning and migration for BLACKICE 3.0.

Provides schema evolution support:
- Version tracking for events
- Migration strategies (up/down)
- Schema compatibility checks
- Automatic migration on load
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeAlias

from blackice.primitives.types import EventType


class SchemaVersion(Enum):
    """Known schema versions."""

    V1 = "1.0.0"  # Initial version
    V2 = "2.0.0"  # Added correlation_id, task_id fields

    @classmethod
    def current(cls) -> SchemaVersion:
        """Get the current schema version."""
        return cls.V2

    @classmethod
    def from_string(cls, version: str) -> SchemaVersion:
        """Parse version from string."""
        for v in cls:
            if v.value == version:
                return v
        raise ValueError(f"Unknown schema version: {version}")


# Type alias for migration functions
MigrationFn: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Migration:
    """A single schema migration.

    Attributes:
        from_version: Source version
        to_version: Target version
        up: Function to migrate forward
        down: Function to migrate backward (optional)
        description: Human-readable description
    """

    from_version: SchemaVersion
    to_version: SchemaVersion
    up: MigrationFn
    down: MigrationFn | None = None
    description: str = ""


class MigrationRegistry:
    """Registry of available migrations."""

    def __init__(self) -> None:
        """Initialize the migration registry."""
        self._migrations: list[Migration] = []
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # V1 -> V2: Add correlation_id and task_id fields
        self.register(
            Migration(
                from_version=SchemaVersion.V1,
                to_version=SchemaVersion.V2,
                up=self._migrate_v1_to_v2,
                down=self._migrate_v2_to_v1,
                description="Add correlation_id and task_id fields",
            )
        )

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations.append(migration)

    def get_migration_path(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> list[Migration]:
        """Get the sequence of migrations needed.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of migrations to apply in order

        Raises:
            ValueError: If no migration path exists
        """
        if from_version == to_version:
            return []

        # Build version graph
        forward = from_version.value < to_version.value

        path: list[Migration] = []
        current = from_version

        while current != to_version:
            found = False
            for migration in self._migrations:
                if forward and migration.from_version == current:
                    path.append(migration)
                    current = migration.to_version
                    found = True
                    break
                elif not forward and migration.to_version == current:
                    if migration.down is None:
                        raise ValueError(
                            f"No downgrade path from {current.value} to {to_version.value}"
                        )
                    path.append(migration)
                    current = migration.from_version
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"No migration path from {from_version.value} to {to_version.value}"
                )

        return path

    @staticmethod
    def _migrate_v1_to_v2(event: dict[str, Any]) -> dict[str, Any]:
        """Migrate event from V1 to V2."""
        result = event.copy()
        result["schema_version"] = SchemaVersion.V2.value

        # Add missing fields with defaults
        if "correlation_id" not in result:
            result["correlation_id"] = None
        if "task_id" not in result:
            result["task_id"] = None
        if "agent_id" not in result:
            result["agent_id"] = None

        return result

    @staticmethod
    def _migrate_v2_to_v1(event: dict[str, Any]) -> dict[str, Any]:
        """Migrate event from V2 to V1 (downgrade)."""
        result = event.copy()
        result["schema_version"] = SchemaVersion.V1.value

        # Remove V2-only fields
        result.pop("correlation_id", None)
        result.pop("task_id", None)
        result.pop("agent_id", None)

        return result


@dataclass
class SchemaValidator:
    """Validates event data against schema requirements."""

    version: SchemaVersion

    def validate(self, event: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate event data.

        Args:
            event: Event data to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # Required fields for all versions
        required_fields = ["id", "run_id", "type", "timestamp", "sequence"]
        for field_name in required_fields:
            if field_name not in event:
                errors.append(f"Missing required field: {field_name}")

        # Version-specific validation
        if self.version == SchemaVersion.V2:
            # V2 requires hash chain fields
            if "hash" not in event:
                errors.append("Missing required field: hash")

        # Type validation
        if "type" in event:
            try:
                EventType(event["type"])
            except ValueError:
                errors.append(f"Invalid event type: {event['type']}")

        # Sequence must be non-negative
        if "sequence" in event and event["sequence"] < 0:
            errors.append(f"Sequence must be non-negative: {event['sequence']}")

        return len(errors) == 0, errors


class EventMigrator:
    """Migrates events between schema versions."""

    def __init__(self, registry: MigrationRegistry | None = None) -> None:
        """Initialize the migrator.

        Args:
            registry: Migration registry, uses default if not provided
        """
        self.registry = registry or MigrationRegistry()

    def detect_version(self, event: dict[str, Any]) -> SchemaVersion:
        """Detect the schema version of an event.

        Args:
            event: Event data

        Returns:
            Detected schema version
        """
        # Check explicit version field
        if "schema_version" in event:
            return SchemaVersion.from_string(event["schema_version"])

        # Heuristic detection based on fields present
        if "correlation_id" in event or "task_id" in event or "agent_id" in event:
            return SchemaVersion.V2

        return SchemaVersion.V1

    def migrate(
        self,
        event: dict[str, Any],
        target_version: SchemaVersion | None = None,
    ) -> dict[str, Any]:
        """Migrate an event to the target version.

        Args:
            event: Event data to migrate
            target_version: Target version, defaults to current

        Returns:
            Migrated event data
        """
        target = target_version or SchemaVersion.current()
        current = self.detect_version(event)

        if current == target:
            return event

        path = self.registry.get_migration_path(current, target)

        result = event.copy()
        for migration in path:
            if migration.from_version.value < migration.to_version.value:
                # Forward migration
                result = migration.up(result)
            else:
                # Backward migration
                if migration.down is None:
                    raise ValueError(
                        f"Cannot downgrade from {migration.from_version.value}"
                    )
                result = migration.down(result)

        return result

    def migrate_batch(
        self,
        events: list[dict[str, Any]],
        target_version: SchemaVersion | None = None,
    ) -> list[dict[str, Any]]:
        """Migrate a batch of events.

        Args:
            events: List of events to migrate
            target_version: Target version, defaults to current

        Returns:
            List of migrated events
        """
        return [self.migrate(event, target_version) for event in events]


@dataclass
class SchemaInfo:
    """Information about event schema.

    Attributes:
        version: Current schema version
        event_count: Number of events at this version
        first_event_at: Timestamp of first event
        last_event_at: Timestamp of last event
        needs_migration: Whether migration is needed
    """

    version: SchemaVersion
    event_count: int = 0
    first_event_at: datetime | None = None
    last_event_at: datetime | None = None
    needs_migration: bool = False

    @property
    def is_current(self) -> bool:
        """Check if schema is at current version."""
        return self.version == SchemaVersion.current()


def add_schema_version(event: dict[str, Any]) -> dict[str, Any]:
    """Add schema version to event if missing.

    Args:
        event: Event data

    Returns:
        Event with schema_version field
    """
    if "schema_version" not in event:
        result = event.copy()
        result["schema_version"] = SchemaVersion.current().value
        return result
    return event
