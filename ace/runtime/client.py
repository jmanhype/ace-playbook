"""Runtime client for orchestrating online updates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ace.curator.curator_models import CuratorInsight, CuratorOutput
from ace.curator.curator_service import CuratorService


class ReflectorProgram(Protocol):
    """Protocol that reflectors must implement for online updates."""

    def generate_insights(
        self,
        *,
        task_id: str,
        domain_id: str,
        prediction: Mapping[str, Any],
        feedback: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return structured insights ready for curation."""


class OnlineUpdatePayload(BaseModel):
    """Bundled example, prediction, and feedback for online updates."""

    task_id: str
    domain_id: str
    prediction: Dict[str, Any]
    feedback: Dict[str, Any]
    insights: List[CuratorInsight] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OnlineUpdateResponse(BaseModel):
    """Curator delta returned from an online update."""

    task_id: str
    domain_id: str
    curator_output: CuratorOutput

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def delta(self):
        return self.curator_output.delta


class RuntimeClient:
    """Thin facade that wires reflector and curator into a single call."""

    def __init__(
        self,
        *,
        curator_service: CuratorService,
        reflector: Optional[ReflectorProgram] = None,
    ) -> None:
        self.curator_service = curator_service
        self.reflector = reflector

    def apply_online_update(self, payload: OnlineUpdatePayload) -> OnlineUpdateResponse:
        """Run reflector + curator once and return the resulting delta."""

        insights: List[CuratorInsight] = list(payload.insights)

        if not insights:
            if self.reflector is None:
                raise ValueError("No reflector configured and no insights provided")
            raw_insights = self.reflector.generate_insights(
                task_id=payload.task_id,
                domain_id=payload.domain_id,
                prediction=payload.prediction,
                feedback=payload.feedback,
            )
            insights = [CuratorInsight.model_validate(item) for item in raw_insights]

        curator_output = self.curator_service.merge_insights(
            task_id=payload.task_id,
            domain_id=payload.domain_id,
            insights=[insight.model_dump() for insight in insights],
        )

        return OnlineUpdateResponse(
            task_id=payload.task_id,
            domain_id=payload.domain_id,
            curator_output=curator_output,
        )
