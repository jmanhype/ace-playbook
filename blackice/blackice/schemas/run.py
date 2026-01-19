"""Run schema for BLACKICE 3.0.

The Run represents a single execution of the agentic software factory,
from initial vision to completed software artifact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import (
    Edition,
    RunId,
    RunStatus,
    Timestamp,
    new_run_id,
)


class RunConfig(BaseModel):
    """Configuration for a BLACKICE run."""

    # Model configuration
    model_provider: str = Field(default="claude", description="Primary model provider")
    model_name: str | None = Field(default=None, description="Specific model name")

    # Execution configuration
    max_tokens: int = Field(default=100_000, ge=1000, description="Token budget for run")
    max_cost_dollars: float = Field(default=10.0, ge=0.0, description="Cost budget in USD")
    timeout_seconds: int = Field(default=3600, ge=60, description="Run timeout")
    max_task_retries: int = Field(default=3, ge=0, le=10, description="Max retries per task")

    # Behavior configuration
    consensus_voting: bool = Field(default=False, description="Enable multi-agent consensus")
    parallel_tasks: int = Field(default=1, ge=1, le=10, description="Max parallel tasks")

    # Recovery configuration
    checkpoint_interval: int = Field(default=300, ge=60, description="Seconds between checkpoints")
    enable_recovery: bool = Field(default=True, description="Enable crash recovery (Core+)")

    # Enterprise configuration
    taskspec_id: str | None = Field(default=None, description="TaskSpec ID (Enterprise)")
    generate_receipt: bool = Field(default=False, description="Generate verifiable receipt")


class Run(BaseModel):
    """A BLACKICE run representing a vision-to-software execution.

    The Run is the primary entity that tracks the entire lifecycle
    of transforming a user's vision into working software.

    Attributes:
        id: Unique identifier for this run
        vision: User's natural language description of desired software
        status: Current status of the run
        edition: BLACKICE edition (Lite, Core, Enterprise)
        created_at: When the run was created
        completed_at: When the run completed (success or failure)
        workspace_path: Path to the run's workspace directory
        config: Run configuration options
        metadata: Additional metadata for tracking
    """

    id: RunId = Field(default_factory=new_run_id)
    vision: str = Field(..., min_length=10, max_length=10_000, description="Vision description")
    status: RunStatus = Field(default=RunStatus.PENDING)
    edition: Edition = Field(default=Edition.LITE)

    created_at: Timestamp = Field(default_factory=Timestamp.now)
    completed_at: Timestamp | None = Field(default=None)

    workspace_path: Path | None = Field(default=None, description="Run workspace directory")
    config: RunConfig = Field(default_factory=RunConfig)

    # Tracking
    current_phase: str | None = Field(default=None, description="Current execution phase")
    task_count: int = Field(default=0, ge=0, description="Total tasks in run")
    completed_tasks: int = Field(default=0, ge=0, description="Completed task count")
    failed_tasks: int = Field(default=0, ge=0, description="Failed task count")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Links
    taskspec_id: str | None = Field(default=None, description="Associated TaskSpec (Enterprise)")
    receipt_id: str | None = Field(default=None, description="Generated receipt ID (Enterprise)")

    class Config:
        """Pydantic configuration."""

        frozen = False  # Mutable during execution

    @property
    def is_active(self) -> bool:
        """Check if run is still active (not terminal)."""
        return self.status in (RunStatus.PENDING, RunStatus.RUNNING, RunStatus.PAUSED)

    @property
    def is_terminal(self) -> bool:
        """Check if run has reached a terminal state."""
        return self.status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)

    @property
    def progress(self) -> float:
        """Calculate run progress as a percentage."""
        if self.task_count == 0:
            return 0.0
        return (self.completed_tasks / self.task_count) * 100

    def start(self) -> Run:
        """Mark run as started."""
        self.status = RunStatus.RUNNING
        return self

    def pause(self) -> Run:
        """Mark run as paused."""
        self.status = RunStatus.PAUSED
        return self

    def resume(self) -> Run:
        """Resume a paused run."""
        self.status = RunStatus.RUNNING
        return self

    def complete(self) -> Run:
        """Mark run as successfully completed."""
        self.status = RunStatus.SUCCEEDED
        self.completed_at = Timestamp.now()
        return self

    def fail(self, reason: str | None = None) -> Run:
        """Mark run as failed."""
        self.status = RunStatus.FAILED
        self.completed_at = Timestamp.now()
        if reason:
            self.metadata["failure_reason"] = reason
        return self

    def cancel(self) -> Run:
        """Mark run as cancelled."""
        self.status = RunStatus.CANCELLED
        self.completed_at = Timestamp.now()
        return self


class RunSummary(BaseModel):
    """Lightweight run summary for listings."""

    id: RunId
    vision_preview: str = Field(..., max_length=100)
    status: RunStatus
    edition: Edition
    created_at: Timestamp
    progress: float = Field(ge=0.0, le=100.0)

    @classmethod
    def from_run(cls, run: Run) -> RunSummary:
        """Create summary from full Run."""
        return cls(
            id=run.id,
            vision_preview=run.vision[:97] + "..." if len(run.vision) > 100 else run.vision,
            status=run.status,
            edition=run.edition,
            created_at=run.created_at,
            progress=run.progress,
        )
