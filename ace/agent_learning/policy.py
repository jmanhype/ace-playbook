"""Exploration/exploitation policies for the live loop."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Mapping

from ace.agent_learning.types import FeedbackPacket, TaskSpecification, WorldModelPrediction


class BasePolicy(ABC):
    """Base interface for deciding whether to apply curator updates."""

    @abstractmethod
    def should_update(
        self,
        *,
        task: TaskSpecification,
        prediction: WorldModelPrediction,
        feedback: FeedbackPacket,
        metrics: Mapping[str, float],
    ) -> bool:
        """Return ``True`` when the curator should receive an update."""


class EpsilonGreedyPolicy(BasePolicy):
    """Simple policy that encourages occasional exploration."""

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        metric: str = "accuracy",
        threshold: float = 0.7,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be within [0, 1]")
        self._epsilon = epsilon
        self._metric = metric
        self._threshold = threshold
        self._rng = rng or random.Random()

    def should_update(
        self,
        *,
        task: TaskSpecification,
        prediction: WorldModelPrediction,
        feedback: FeedbackPacket,
        metrics: Mapping[str, float],
    ) -> bool:
        del task, prediction, feedback  # Currently unused but kept for extensibility
        if self._rng.random() < self._epsilon:
            return True
        score = metrics.get(self._metric, 0.0)
        return score < self._threshold


__all__ = ["BasePolicy", "EpsilonGreedyPolicy"]
