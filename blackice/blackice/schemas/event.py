"""Event schema for BLACKICE 3.0.

Events form an append-only log of all actions taken during a run,
enabling recovery, auditing, and debugging.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, computed_field

from blackice.primitives.types import (
    CorrelationId,
    EventId,
    EventType,
    Hash,
    RunId,
    TaskId,
    Timestamp,
    new_event_id,
)


class Event(BaseModel):
    """An immutable event in the run's event log.

    Events form a hash-chained append-only log that provides:
    - Full audit trail of all actions
    - Recovery capability for crash recovery (Core+)
    - Verification for compliance (Enterprise)

    Attributes:
        id: Unique identifier for this event
        run_id: Run this event belongs to
        type: Type of event
        payload: Event-specific data
        timestamp: When the event occurred
        correlation_id: ID for distributed tracing
        sequence: Position in the event log
        previous_hash: Hash of the previous event
        hash: Hash of this event (computed)
    """

    id: EventId = Field(default_factory=new_event_id)
    run_id: RunId
    type: EventType

    # Event data
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: Timestamp = Field(default_factory=Timestamp.now)

    # Correlation
    correlation_id: str | None = Field(default=None)
    task_id: TaskId | None = Field(default=None)
    agent_id: str | None = Field(default=None)

    # Sequencing (for ordering and hash chain)
    sequence: int = Field(ge=0)
    previous_hash: str | None = Field(default=None)

    class Config:
        """Pydantic configuration."""

        frozen = True  # Events are immutable

    @computed_field  # type: ignore[misc]
    @property
    def hash(self) -> str:
        """Compute SHA-256 hash of the event for chain integrity."""
        # Create deterministic string representation
        data = f"{self.id}:{self.run_id}:{self.type.value}:{self.sequence}:{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_chain(self, expected_previous_hash: str | None) -> bool:
        """Verify this event's previous hash matches expected."""
        return self.previous_hash == expected_previous_hash


class EventPayloads:
    """Standard payload structures for different event types."""

    @staticmethod
    def run_started(vision: str, edition: str, config: dict[str, Any]) -> dict[str, Any]:
        """Payload for RUN_STARTED event."""
        return {
            "vision_preview": vision[:200] if len(vision) > 200 else vision,
            "edition": edition,
            "config": config,
        }

    @staticmethod
    def run_completed(
        duration_seconds: float, task_count: int, artifact_count: int
    ) -> dict[str, Any]:
        """Payload for RUN_COMPLETED event."""
        return {
            "duration_seconds": duration_seconds,
            "task_count": task_count,
            "artifact_count": artifact_count,
        }

    @staticmethod
    def run_failed(error: str, task_id: str | None = None) -> dict[str, Any]:
        """Payload for RUN_FAILED event."""
        return {
            "error": error[:500] if len(error) > 500 else error,
            "failing_task_id": task_id,
        }

    @staticmethod
    def task_started(task_name: str, attempt: int) -> dict[str, Any]:
        """Payload for TASK_STARTED event."""
        return {"task_name": task_name, "attempt": attempt}

    @staticmethod
    def task_completed(task_name: str, duration_seconds: float) -> dict[str, Any]:
        """Payload for TASK_COMPLETED event."""
        return {"task_name": task_name, "duration_seconds": duration_seconds}

    @staticmethod
    def task_failed(task_name: str, error: str, attempt: int) -> dict[str, Any]:
        """Payload for TASK_FAILED event."""
        return {
            "task_name": task_name,
            "error": error[:500] if len(error) > 500 else error,
            "attempt": attempt,
        }

    @staticmethod
    def command_executed(
        command: str, exit_code: int, duration_seconds: float
    ) -> dict[str, Any]:
        """Payload for COMMAND_EXECUTED event."""
        return {
            "command_preview": command[:200] if len(command) > 200 else command,
            "exit_code": exit_code,
            "duration_seconds": duration_seconds,
        }

    @staticmethod
    def command_blocked(command: str, reason: str, policy: str) -> dict[str, Any]:
        """Payload for COMMAND_BLOCKED event."""
        return {
            "command_preview": command[:100] if len(command) > 100 else command,
            "reason": reason,
            "policy": policy,
        }

    @staticmethod
    def agent_spawned(agent_id: str, role: str, model: str) -> dict[str, Any]:
        """Payload for AGENT_SPAWNED event."""
        return {"agent_id": agent_id, "role": role, "model": model}

    @staticmethod
    def consensus_reached(
        decision: str, votes: dict[str, str], agreement_ratio: float
    ) -> dict[str, Any]:
        """Payload for CONSENSUS_REACHED event."""
        return {
            "decision": decision,
            "votes": votes,
            "agreement_ratio": agreement_ratio,
        }

    @staticmethod
    def checkpoint_created(checkpoint_id: str, event_sequence: int) -> dict[str, Any]:
        """Payload for CHECKPOINT_CREATED event."""
        return {
            "checkpoint_id": checkpoint_id,
            "event_sequence": event_sequence,
        }


class EventLog(BaseModel):
    """Append-only event log for a run.

    Provides hash-chain integrity and ordered event access.
    """

    run_id: RunId
    events: list[Event] = Field(default_factory=list)

    def append(self, event_type: EventType, payload: dict[str, Any], **kwargs: Any) -> Event:
        """Append a new event to the log.

        Creates the event with proper sequencing and hash chain.
        """
        sequence = len(self.events)
        previous_hash = self.events[-1].hash if self.events else None

        event = Event(
            run_id=self.run_id,
            type=event_type,
            payload=payload,
            sequence=sequence,
            previous_hash=previous_hash,
            **kwargs,
        )
        self.events.append(event)
        return event

    def verify_integrity(self) -> tuple[bool, str | None]:
        """Verify the hash chain integrity.

        Returns (is_valid, error_message).
        """
        previous_hash: str | None = None

        for i, event in enumerate(self.events):
            if event.sequence != i:
                return False, f"Sequence mismatch at index {i}"
            if event.previous_hash != previous_hash:
                return False, f"Hash chain broken at sequence {i}"
            previous_hash = event.hash

        return True, None

    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]

    def get_events_for_task(self, task_id: TaskId) -> list[Event]:
        """Get all events for a specific task."""
        return [e for e in self.events if e.task_id == task_id]

    def get_latest_event(self) -> Event | None:
        """Get the most recent event."""
        return self.events[-1] if self.events else None

    @property
    def latest_hash(self) -> str | None:
        """Get the hash of the most recent event."""
        return self.events[-1].hash if self.events else None
