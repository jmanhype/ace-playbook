"""Utility datasets for seeding the agent live loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

from ace.agent_learning.types import TaskSpecification


@dataclass
class StaticTaskDataset:
    """Simple iterator over an in-memory sequence of tasks."""

    tasks: Sequence[TaskSpecification]

    def __iter__(self) -> Iterator[TaskSpecification]:
        return iter(self.tasks)

    def cycle(self) -> Iterable[TaskSpecification]:
        """Yield tasks in order indefinitely."""

        while True:
            for task in self.tasks:
                yield task


__all__ = ["StaticTaskDataset"]
