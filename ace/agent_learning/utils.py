"""Helper utilities for composing the live loop in tests and demos."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from ace.runtime.client import RuntimeClient
from ace.runtime.metrics import MetricRegistry, accuracy_metric
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="agent_utils")


@dataclass
class InMemoryCuratorOperation:
    """Lightweight stand-in for :class:`ace.curator.curator_models.CuratorOperation`."""

    type: str
    section: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "type": self.type,
            "section": self.section,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass
class InMemoryCuratorDelta:
    operations: List[InMemoryCuratorOperation] = field(default_factory=list)


@dataclass
class InMemoryCuratorOutput:
    task_id: str
    domain_id: str
    delta: InMemoryCuratorDelta
    delta_updates: List[Dict[str, Any]]
    updated_playbook: List[Dict[str, Any]]
    new_bullets_added: int
    existing_bullets_incremented: int
    duplicates_detected: int
    bullets_quarantined: int
    bullets_promoted: int


class InMemoryCuratorService:
    """Minimal curator facade that stores insights in process memory."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        self._counter: MutableMapping[str, int] = defaultdict(int)

    def merge_insights(
        self,
        *,
        task_id: str,
        domain_id: str,
        insights: Iterable[Mapping[str, Any]],
        **_: object,
    ) -> InMemoryCuratorOutput:
        operations: List[InMemoryCuratorOperation] = []
        delta_updates: List[Dict[str, Any]] = []
        stored_insights = self._store[domain_id]
        for raw in insights:
            content = str(raw.get("content", "")).strip()
            if not content:
                logger.debug("in_memory_curator_skip_empty", domain_id=domain_id)
                continue
            section_value = raw.get("section", "Helpful")
            if hasattr(section_value, "value"):
                section = str(section_value.value)
            else:
                section = str(section_value)
            tags = list(raw.get("tags", []))
            record = {"content": content, "section": section, "tags": tags}
            stored_insights.append(record)
            bullet_idx = self._counter[domain_id]
            self._counter[domain_id] += 1
            operations.append(
                InMemoryCuratorOperation(
                    type="add",
                    section=section,
                    content=content,
                    metadata={"tags": tags},
                )
            )
            delta_updates.append(
                {
                    "operation": "add",
                    "bullet_id": f"{domain_id}-{bullet_idx}",
                    "metadata": {"tags": tags, "content": content},
                }
            )
        delta = InMemoryCuratorDelta(operations=operations)
        output = InMemoryCuratorOutput(
            task_id=task_id,
            domain_id=domain_id,
            delta_updates=delta_updates,
            updated_playbook=list(stored_insights),
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
