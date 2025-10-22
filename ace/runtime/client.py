"""Runtime client for orchestrating online updates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field


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
    insights: List[Any] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OnlineUpdateResponse(BaseModel):
    """Curator delta returned from an online update."""

    task_id: str
    domain_id: str
    curator_output: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def delta(self):
        return self.curator_output.delta


class CuratorServiceProtocol(Protocol):
    """Duck-typed subset of :class:`CuratorService` used by the runtime client."""

    def merge_insights(
        self,
        *,
        task_id: str,
        domain_id: str,
        insights: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Merge insights into the curator and return a delta object."""


class RuntimeClient:
    """Thin facade that wires reflector and curator into a single call."""

    def __init__(
        self,
        *,
        curator_service: CuratorServiceProtocol,
        reflector: Optional[ReflectorProgram] = None,
    ) -> None:
        self.curator_service = curator_service
        self.reflector = reflector

    def apply_online_update(self, payload: OnlineUpdatePayload) -> OnlineUpdateResponse:
        """Run reflector + curator once and return the resulting delta."""

        insights_raw: List[Any] = list(payload.insights)

        if not insights_raw:
            if self.reflector is None:
                raise ValueError("No reflector configured and no insights provided")
            raw_insights = self.reflector.generate_insights(
                task_id=payload.task_id,
                domain_id=payload.domain_id,
                prediction=payload.prediction,
                feedback=payload.feedback,
            )
            insights_raw = list(raw_insights)

        normalized_insights: List[Mapping[str, Any]] = []
        for insight in insights_raw:
            if hasattr(insight, "model_dump"):
                normalized_insights.append(insight.model_dump())
            elif isinstance(insight, Mapping):
                normalized_insights.append(dict(insight))
            else:
                raise TypeError(
                    "Insights must be mappings or pydantic models with model_dump()",
                )

        curator_output = self.curator_service.merge_insights(
            task_id=payload.task_id,
            domain_id=payload.domain_id,
            insights=normalized_insights,
        )

        return OnlineUpdateResponse(
            task_id=payload.task_id,
            domain_id=payload.domain_id,
            curator_output=curator_output,
        )
