"""World model adapter that produces structured task predictions."""

from __future__ import annotations

from textwrap import dedent
from typing import Optional

from ace.agent_learning.types import FeedbackPacket, TaskSpecification, WorldModelPrediction
from ace.llm_client import BaseLLMClient
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_world_model")


class WorldModel:
    """Thin wrapper around an :class:`~ace.llm_client.BaseLLMClient`."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._temperature = temperature

    def predict(self, task: TaskSpecification) -> WorldModelPrediction:
        """Run the model and return a structured prediction."""

        prompt = self._build_prompt(task)
        prediction = self._client.structured_completion(
            prompt=prompt,
            response_model=WorldModelPrediction,
            model=self._model,
            temperature=self._temperature,
            metadata={"task_id": task.task_id, "response_key": "WorldModelPrediction"},
        )
        logger.debug(
            "world_model_prediction", task_id=task.task_id, answer=prediction.answer, confidence=prediction.confidence
        )
        return prediction

    def derive_feedback(
        self, task: TaskSpecification, prediction: WorldModelPrediction
    ) -> FeedbackPacket:
        """Generate a lightweight feedback packet for metric evaluation."""

        evaluation = "unknown"
        if task.ground_truth is not None:
            is_correct = prediction.answer.strip().lower() == task.ground_truth.strip().lower()
            evaluation = "correct" if is_correct else "incorrect"
        feedback = FeedbackPacket(ground_truth=task.ground_truth, evaluation=evaluation)
        logger.debug("world_model_feedback", task_id=task.task_id, evaluation=evaluation)
        return feedback

    @staticmethod
    def _build_prompt(task: TaskSpecification) -> str:
        """Create a deterministic prompt for structured reasoning."""

        return dedent(
            f"""
            You are an autonomous software agent tasked with solving problems using
            the ACE playbook.  Produce a JSON object with the following keys:
            - answer: string containing your final answer
            - reasoning: ordered list of short steps showing your reasoning
            - confidence: float between 0 and 1 representing confidence
            - raw_response: optional metadata dictionary

            Problem description: {task.description}
            Domain: {task.domain_id}
            """
        ).strip()


__all__ = ["WorldModel"]
