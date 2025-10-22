"""Shared pydantic models used across the agent learning package."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskSpecification(BaseModel):
    """Description of a task that the agent can attempt."""

    task_id: str = Field(..., description="Unique identifier for the task")
    domain_id: str = Field(..., description="Playbook domain namespace")
    description: str = Field(..., description="Problem statement presented to the agent")
    ground_truth: Optional[str] = Field(
        default=None, description="Optional answer for automatic reward shaping"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class WorldModelPrediction(BaseModel):
    """Structured output produced by the world model."""

    answer: str = Field(..., description="Primary answer emitted by the generator")
    reasoning: List[str] = Field(default_factory=list, description="Ordered reasoning trace")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Self-reported confidence score"
    )
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Vendor specific payload for debugging"
    )


class FeedbackPacket(BaseModel):
    """Feedback derived from evaluation metrics or ground-truth annotations."""

    ground_truth: Optional[str] = Field(default=None, description="Known correct answer")
    evaluation: str = Field(default="unknown", description="Short label describing the result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class EpisodeResult(BaseModel):
    """Container returned by the live loop for every completed task."""

    task_id: str
    domain_id: str
    prediction: WorldModelPrediction
    feedback: FeedbackPacket
    metrics: Dict[str, float] = Field(default_factory=dict)
    curator_operations: List[Any] = Field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary for logging or analytics."""

        operations: List[Dict[str, Any]] = []
        for op in self.curator_operations:
            if hasattr(op, "model_dump"):
                operations.append(op.model_dump())
            elif isinstance(op, dict):
                operations.append(op)
            else:
                operations.append(getattr(op, "__dict__", {}))
        return {
            "task_id": self.task_id,
            "domain_id": self.domain_id,
            "answer": self.prediction.answer,
            "metrics": self.metrics,
            "operations": operations,
        }


__all__ = [
    "EpisodeResult",
    "FeedbackPacket",
    "TaskSpecification",
    "WorldModelPrediction",
]
