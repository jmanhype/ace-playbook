"""Basic guardrails to keep live loop predictions in check."""

from __future__ import annotations

from ace.agent_learning.types import TaskSpecification, WorldModelPrediction
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_guardrails")


class PredictionGuardrails:
    """Validation helpers that raise descriptive errors for invalid predictions."""

    def __init__(self, *, require_reasoning: bool = True) -> None:
        self._require_reasoning = require_reasoning

    def validate(self, task: TaskSpecification, prediction: WorldModelPrediction) -> None:
        if not prediction.answer.strip():
            raise ValueError(f"Prediction for task {task.task_id} produced an empty answer")
        if self._require_reasoning and not prediction.reasoning:
            raise ValueError(f"Prediction for task {task.task_id} is missing reasoning trace")
        logger.debug("prediction_validated", task_id=task.task_id)


__all__ = ["PredictionGuardrails"]
