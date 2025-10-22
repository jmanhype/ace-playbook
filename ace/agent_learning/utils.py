"""Helper utilities for composing the live loop in tests and demos."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Mapping, MutableMapping

from ace.curator.curator_models import (
    CuratorDelta,
    CuratorInsight,
    CuratorOperation,
    CuratorOperationType,
    CuratorOutput,
    DeltaUpdate,
)
from ace.runtime.client import RuntimeClient
from ace.runtime.metrics import MetricRegistry, accuracy_metric
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_utils")


class InMemoryCuratorService:
    """Minimal curator facade that stores insights in process memory."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, List[CuratorInsight]] = defaultdict(list)
        self._counter: MutableMapping[str, int] = defaultdict(int)

    def merge_insights(
        self,
        *,
        task_id: str,
        domain_id: str,
        insights: List[Mapping[str, object]],
        **_: object,
    ) -> CuratorOutput:
        operations: List[CuratorOperation] = []
        delta_updates: List[DeltaUpdate] = []
        stored_insights = self._store[domain_id]
        for raw in insights:
            insight = CuratorInsight.model_validate(raw)
            stored_insights.append(insight)
            bullet_idx = self._counter[domain_id]
            self._counter[domain_id] += 1
            operations.append(
                CuratorOperation(
                    type=CuratorOperationType.ADD,
                    section=insight.section,
                    content=insight.content,
                    metadata={"tags": insight.tags},
                )
            )
            delta_updates.append(
                DeltaUpdate(
                    operation=CuratorOperationType.ADD,
                    bullet_id=f"{domain_id}-{bullet_idx}",
                    metadata={"tags": insight.tags, "content": insight.content},
                )
            )
        delta = CuratorDelta(operations=operations)
        output = CuratorOutput(
            task_id=task_id,
            domain_id=domain_id,
            delta_updates=delta_updates,
            updated_playbook=[],
            new_bullets_added=len(operations),
            existing_bullets_incremented=0,
            duplicates_detected=0,
            bullets_quarantined=0,
            bullets_promoted=0,
            delta=delta,
        )
        logger.debug(
            "in_memory_curator_merge",
            task_id=task_id,
            domain_id=domain_id,
            new_bullets=output.new_bullets_added,
        )
        return output


def create_in_memory_runtime(
    *,
    reflector=None,
) -> RuntimeClient:
    """Return a :class:`RuntimeClient` wired with :class:`InMemoryCuratorService`."""

    curator_service = InMemoryCuratorService()
    return RuntimeClient(curator_service=curator_service, reflector=reflector)


def prepare_default_metrics() -> MetricRegistry:
    """Return a registry with the default accuracy metric pre-registered."""
    registry = MetricRegistry()
    registry.register("accuracy", accuracy_metric)
    return registry


__all__ = ["InMemoryCuratorService", "create_in_memory_runtime", "prepare_default_metrics"]
