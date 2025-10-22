"""Metric hook definitions for runtime optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Protocol


class MetricFunction(Protocol):
    """Callable that computes a scalar score for a task invocation."""

    def __call__(
        self,
        *,
        task_id: str,
        prediction: Mapping[str, Any],
        feedback: Mapping[str, Any],
    ) -> float:
        """Compute a score in [0, 1] where higher is better."""


@dataclass
class MetricResult:
    """Outcome of a metric evaluation."""

    name: str
    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MetricRegistry:
    """Registry that keeps metric functions pluggable at runtime."""

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricFunction] = {}

    def register(self, name: str, fn: MetricFunction) -> None:
        """Register or overwrite a metric by name."""

        self._metrics[name] = fn

    def evaluate(
        self,
        name: str,
        *,
        task_id: str,
        prediction: Mapping[str, Any],
        feedback: Mapping[str, Any],
    ) -> MetricResult:
        """Evaluate a registered metric."""

        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        value = self._metrics[name](
            task_id=task_id,
            prediction=prediction,
            feedback=feedback,
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Metric '{name}' returned {value}, expected value within [0, 1]"
            )
        return MetricResult(name=name, value=value)

    def evaluate_all(
        self,
        *,
        task_id: str,
        prediction: Mapping[str, Any],
        feedback: Mapping[str, Any],
    ) -> Iterable[MetricResult]:
        """Evaluate all registered metrics for the given invocation."""

        for name, fn in self._metrics.items():
            yield MetricResult(
                name=name,
                value=fn(task_id=task_id, prediction=prediction, feedback=feedback),
            )

    def wrap(self, fn: MetricFunction) -> MetricFunction:
        """Return the metric function for decorators or dependency injection."""

        return fn


def accuracy_metric(
    *, task_id: str, prediction: Mapping[str, Any], feedback: Mapping[str, Any]
) -> float:
    """Default accuracy metric that checks ground-truth equality."""

    expected = feedback.get("ground_truth")
    actual = prediction.get("answer")
    return 1.0 if expected is not None and expected == actual else 0.0


DEFAULT_REGISTRY = MetricRegistry()
DEFAULT_REGISTRY.register("accuracy", accuracy_metric)
