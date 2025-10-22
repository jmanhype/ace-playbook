"""Reflection utilities that emit structured insights for the curator."""

from __future__ import annotations

import json
from enum import Enum
from textwrap import dedent
from typing import List, Optional

from pydantic import BaseModel, Field

from ace.llm_client import BaseLLMClient
from ace.runtime.client import ReflectorProgram
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_reflection")


class InsightSection(str, Enum):
    """Local copy of curator insight sections used for lightweight deps."""

    HELPFUL = "Helpful"
    HARMFUL = "Harmful"
    NEUTRAL = "Neutral"


class InsightPayload(BaseModel):
    """Structured insight emitted by the reflection engine."""

    content: str = Field(..., description="Observation or recommendation text")
    section: InsightSection = Field(
        default=InsightSection.HELPFUL, description="Playbook section for the insight"
    )
    tags: List[str] = Field(default_factory=list, description="Categorical tags")


class ReflectionResponse(BaseModel):
    """Batch response containing insights."""

    insights: List[InsightPayload] = Field(default_factory=list)


class ReflectionEngine:
    """Adapter that turns task traces into curator-ready insights."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        model: Optional[str] = None,
        max_insights: int = 3,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._max_insights = max_insights

    def generate(
        self,
        *,
        task_id: str,
        domain_id: str,
        prediction: dict,
        feedback: dict,
    ) -> List[InsightPayload]:
        prompt = self._build_prompt(task_id=task_id, domain_id=domain_id, prediction=prediction, feedback=feedback)
        response = self._client.structured_completion(
            prompt=prompt,
            response_model=ReflectionResponse,
            model=self._model,
            metadata={"task_id": task_id, "response_key": "ReflectionResponse"},
        )
        insights = response.insights[: self._max_insights]
        logger.debug("reflection_generated", task_id=task_id, count=len(insights))
        return insights

    @staticmethod
    def _build_prompt(
        *, task_id: str, domain_id: str, prediction: dict, feedback: dict
    ) -> str:
        reasoning_entries = prediction.get("reasoning", []) or []
        reasoning = (
            "\n".join(reasoning_entries) if reasoning_entries else "(not provided)"
        )
        prediction_json = json.dumps(
            prediction,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        feedback_json = json.dumps(
            feedback,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        return dedent(
            f"""
            You are the reflection module in the ACE live loop.  Using the JSON
            schema (insights: list[{{content, section, tags}}]) produce up to three
            insights that summarise what should be added to the playbook.

            Task ID: {task_id}
            Domain: {domain_id}
            Prediction:
            {prediction_json}
            Feedback:
            {feedback_json}
            Reasoning Trace:
            {reasoning}
            """
        ).strip()


class RuntimeReflector(ReflectorProgram):
    """Implementation of :class:`ace.runtime.client.ReflectorProgram`."""

    def __init__(self, engine: ReflectionEngine) -> None:
        self._engine = engine

    def generate_insights(
        self,
        *,
        task_id: str,
        domain_id: str,
        prediction: dict,
        feedback: dict,
    ) -> List[dict]:
        return [
            insight.model_dump()
            for insight in self._engine.generate(
                task_id=task_id,
                domain_id=domain_id,
                prediction=prediction,
                feedback=feedback,
            )
        ]


__all__ = [
    "InsightSection",
    "InsightPayload",
    "ReflectionEngine",
    "RuntimeReflector",
    "ReflectionResponse",
]
