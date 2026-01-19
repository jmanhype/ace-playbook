"""Base types for BLACKICE 3.0.

This module defines fundamental types used throughout the system:
- Identifiers (RunId, TaskId, EventId, AgentId)
- Enumerations (AgentRole, Edition, RunStatus, TaskStatus)
- Value objects (Timestamp, Hash)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import NewType
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, Field, field_validator

# === Identifier Types ===

RunId = NewType("RunId", UUID)
TaskId = NewType("TaskId", UUID)
EventId = NewType("EventId", UUID)
AgentId = NewType("AgentId", str)
BeadId = NewType("BeadId", str)


def new_run_id() -> RunId:
    """Generate a new RunId."""
    return RunId(uuid4())


def new_task_id() -> TaskId:
    """Generate a new TaskId."""
    return TaskId(uuid4())


def new_event_id() -> EventId:
    """Generate a new EventId."""
    return EventId(uuid4())


# === Enumerations ===


class Edition(str, Enum):
    """BLACKICE edition tiers with additive features."""

    LITE = "lite"
    CORE = "core"
    ENTERPRISE = "enterprise"

    def includes(self, other: Edition) -> bool:
        """Check if this edition includes features of another edition."""
        order = {Edition.LITE: 0, Edition.CORE: 1, Edition.ENTERPRISE: 2}
        return order[self] >= order[other]


class AgentRole(str, Enum):
    """Specialist agent roles in the colony."""

    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DOCUMENTER = "documenter"
    SECURITY = "security"
    ORCHESTRATOR = "orchestrator"


class RunStatus(str, Enum):
    """Status of a BLACKICE run."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of an individual task within a run."""

    PENDING = "pending"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class EventType(str, Enum):
    """Types of events in the event log."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    COMMAND_EXECUTED = "command_executed"
    COMMAND_BLOCKED = "command_blocked"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_COMPLETED = "agent_completed"
    CONSENSUS_REACHED = "consensus_reached"
    MEMORY_STORED = "memory_stored"
    MEMORY_RECALLED = "memory_recalled"
    CHECKPOINT_CREATED = "checkpoint_created"
    RECOVERY_STARTED = "recovery_started"


class StrictnessLevel(str, Enum):
    """TaskSpec strictness levels for Enterprise edition.

    Levels (from most to least permissive):
    - LEARNING: Most permissive, allows all deviations with warnings (for training)
    - PERMISSIVE: Warns on deviations but allows them
    - STRICT: Blocks deviations but allows override with acknowledgment
    - LOCKED: Blocks all deviations with no override possible
    """

    LEARNING = "learning"
    PERMISSIVE = "permissive"
    STRICT = "strict"
    LOCKED = "locked"


class PIIPolicy(str, Enum):
    """PII handling policies for receipts and memory storage.

    Policies:
    - RETAIN: Keep original data as-is (for private/internal use)
    - HASH_ONLY: Store only cryptographic hashes (for verification without exposure)
    - REDACT: Remove or mask sensitive data (for shareable receipts)
    - ALLOW: Legacy alias for RETAIN
    - REJECT: Reject storage of data containing PII
    """

    RETAIN = "retain"
    HASH_ONLY = "hash_only"
    REDACT = "redact"
    ALLOW = "allow"  # Legacy alias for RETAIN
    REJECT = "reject"


# === Value Objects ===


class Timestamp(BaseModel):
    """Immutable timestamp with timezone awareness."""

    value: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("value", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamp is UTC."""
        # Handle string input from JSON deserialization
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    def __str__(self) -> str:
        return self.value.isoformat()

    def __hash__(self) -> int:
        return hash(self.value)

    @classmethod
    def now(cls) -> Timestamp:
        """Create a timestamp for the current moment."""
        return cls(value=datetime.now(timezone.utc))


class Hash(BaseModel):
    """Cryptographic hash value (SHA-256)."""

    value: str = Field(..., min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def short(self) -> str:
        """Return first 8 characters of hash for display."""
        return self.value[:8]


class CorrelationId(BaseModel):
    """Correlation ID for distributed tracing.

    Format: {run_id}:{task_id}:{sequence}
    """

    run_id: RunId
    task_id: TaskId | None = None
    sequence: int = 0

    def __str__(self) -> str:
        if self.task_id:
            return f"{self.run_id}:{self.task_id}:{self.sequence}"
        return f"{self.run_id}::{self.sequence}"

    def next(self) -> CorrelationId:
        """Create next correlation ID in sequence."""
        return CorrelationId(
            run_id=self.run_id,
            task_id=self.task_id,
            sequence=self.sequence + 1,
        )

    def with_task(self, task_id: TaskId) -> CorrelationId:
        """Create correlation ID for a specific task."""
        return CorrelationId(
            run_id=self.run_id,
            task_id=task_id,
            sequence=0,
        )


# === JSON Serialization Helpers ===


def serialize_json(obj: BaseModel) -> bytes:
    """Serialize a Pydantic model to JSON bytes using orjson."""
    return orjson.dumps(obj.model_dump(mode="json"))


def deserialize_json(data: bytes, model: type[BaseModel]) -> BaseModel:
    """Deserialize JSON bytes to a Pydantic model."""
    return model.model_validate(orjson.loads(data))
