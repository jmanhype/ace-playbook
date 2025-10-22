"""High-level orchestration loop for Agent Learning."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Mapping, Optional

from ace.agent_learning.exploration import EpisodeResult, ExperienceBuffer
from ace.agent_learning.guardrails import PredictionGuardrails
from ace.agent_learning.policy import BasePolicy
from ace.agent_learning.reflection import RuntimeReflector
from ace.agent_learning.types import FeedbackPacket, TaskSpecification, WorldModelPrediction
from ace.agent_learning.world_model import WorldModel
from ace.runtime.client import OnlineUpdatePayload, RuntimeClient
from ace.runtime.metrics import MetricRegistry
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_live_loop")


class LiveLoop:
    """Runs the end-to-end agent learning flow for a sequence of tasks."""

    def __init__(
        self,
        *,
        runtime_client: RuntimeClient,
        world_model: WorldModel,
        policy: BasePolicy,
        metric_registry: MetricRegistry,
        tasks: Iterable[TaskSpecification],
        experience_buffer: Optional[ExperienceBuffer] = None,
        guardrails: Optional[PredictionGuardrails] = None,
        reflector: Optional[RuntimeReflector] = None,
    ) -> None:
        self.runtime_client = runtime_client
        if reflector is not None:
            self.runtime_client.reflector = reflector
        self.world_model = world_model
        self.policy = policy
        self.metric_registry = metric_registry
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("LiveLoop requires at least one task specification")
        self._task_cycle = itertools.cycle(self.tasks)
        self.experience_buffer = (
            experience_buffer if experience_buffer is not None else ExperienceBuffer()
        )
        self.guardrails = guardrails if guardrails is not None else PredictionGuardrails()

    def run(self, *, episodes: int) -> List[EpisodeResult]:
        """Execute ``episodes`` iterations of the live loop."""

        results: List[EpisodeResult] = []
        for _ in range(episodes):
            task = next(self._task_cycle)
            prediction = self.world_model.predict(task)
            self.guardrails.validate(task, prediction)
            feedback = self.world_model.derive_feedback(task, prediction)
            metrics = self._evaluate_metrics(
                task=task,
                prediction=prediction,
                feedback=feedback,
            )
            should_update = self.policy.should_update(
                task=task,
                prediction=prediction,
                feedback=feedback,
                metrics=metrics,
            )
            curator_operations = []
            if should_update:
                payload = OnlineUpdatePayload(
                    task_id=task.task_id,
                    domain_id=task.domain_id,
                    prediction=prediction.model_dump(),
                    feedback=feedback.model_dump(),
                )
                response = self.runtime_client.apply_online_update(payload)
                curator_operations = list(response.curator_output.delta.operations)
                logger.info(
                    "live_loop_delta_applied",
                    task_id=task.task_id,
                    domain_id=task.domain_id,
                    operations=len(curator_operations),
                )
            else:
                logger.info(
                    "live_loop_skipped_update",
                    task_id=task.task_id,
                    domain_id=task.domain_id,
                    metrics=metrics,
                )
            episode = EpisodeResult(
                task_id=task.task_id,
                domain_id=task.domain_id,
                prediction=prediction,
                feedback=feedback,
                metrics=metrics,
                curator_operations=curator_operations,
            )
            self.experience_buffer.append(episode)
            results.append(episode)
        return results

    def _evaluate_metrics(
        self,
        *,
        task: TaskSpecification,
        prediction: WorldModelPrediction,
        feedback: FeedbackPacket,
    ) -> Mapping[str, float]:
        output = {}
        for result in self.metric_registry.evaluate_all(
            task_id=task.task_id,
            prediction=prediction.model_dump(),
            feedback=feedback.model_dump(),
        ):
            output[result.name] = result.value
        logger.debug("live_loop_metrics", task_id=task.task_id, metrics=output)
        return output


__all__ = ["LiveLoop"]
