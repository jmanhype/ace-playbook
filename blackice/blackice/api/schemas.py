"""API schemas for BLACKICE 3.0.

Pydantic models for request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Run lifecycle states."""

    CREATED = "created"
    RESEARCH = "research"
    PLAN = "plan"
    IMPLEMENT = "implement"
    TEST = "test"
    VERIFY = "verify"
    DELIVER = "deliver"
    FAILED = "failed"
    COMPLETED = "completed"


class ProviderType(str, Enum):
    """Available LLM provider types."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    CLAUDE_MAX = "claude-max"
    OPENAI = "openai"
    ZHIPU = "zhipu"
    ZAI = "z.ai"


class Edition(str, Enum):
    """BLACKICE edition tiers."""

    LITE = "lite"
    CORE = "core"
    ENTERPRISE = "enterprise"


# Request schemas


class RunCreateRequest(BaseModel):
    """Request to create a new run."""

    vision: str = Field(..., description="Natural language description of what to build")
    edition: Edition = Field(default=Edition.CORE, description="BLACKICE edition tier")
    provider: ProviderType = Field(default=ProviderType.OLLAMA, description="LLM provider to use")
    model: str | None = Field(default=None, description="Specific model to use (optional)")
    workspace: str | None = Field(default=None, description="Custom workspace path (optional)")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class RunResumeRequest(BaseModel):
    """Request to resume an existing run."""

    from_phase: str | None = Field(default=None, description="Resume from specific phase")


# Response schemas


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="BLACKICE version")
    providers: dict[str, bool] = Field(default_factory=dict, description="Provider health status")
    latency_ms: float | None = Field(default=None, description="Health check latency")


class ProviderInfo(BaseModel):
    """Information about a provider."""

    name: str
    type: ProviderType
    healthy: bool
    base_url: str | None = None
    model: str | None = None
    latency_ms: float | None = None


class ProvidersResponse(BaseModel):
    """List of available providers."""

    providers: list[ProviderInfo]
    default: ProviderType


class RunSummary(BaseModel):
    """Summary of a run."""

    run_id: str
    vision: str
    status: RunStatus
    edition: Edition
    provider: ProviderType
    created_at: datetime
    updated_at: datetime | None = None
    phase_count: int = 0
    artifact_count: int = 0


class RunDetail(BaseModel):
    """Detailed run information."""

    run_id: str
    vision: str
    status: RunStatus
    edition: Edition
    provider: ProviderType
    model: str | None = None
    workspace: str
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    phases: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    current_phase: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class RunCreateResponse(BaseModel):
    """Response after creating a run."""

    run_id: str
    status: RunStatus
    message: str
    workspace: str


class RunListResponse(BaseModel):
    """List of runs."""

    runs: list[RunSummary]
    total: int
    limit: int
    offset: int


class PhaseResult(BaseModel):
    """Result of a phase execution."""

    phase: str
    status: str
    duration_ms: float
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class RunResultResponse(BaseModel):
    """Final run result."""

    run_id: str
    success: bool
    status: RunStatus
    phases: list[PhaseResult]
    artifacts: list[str]
    total_duration_ms: float
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    run_id: str | None = None


class StreamEvent(BaseModel):
    """Event sent via WebSocket or SSE."""

    event: str
    run_id: str
    phase: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
