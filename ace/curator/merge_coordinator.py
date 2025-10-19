"""Batch merge coordinator for concurrent insight ingestion."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ace.models.playbook import PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="merge_coordinator")


@dataclass
class MergeEvent:
    """Container for high-confidence insights awaiting merge."""

    task_id: str
    insights: List[Dict]
    target_stage: PlaybookStage = PlaybookStage.SHADOW


@dataclass
class DomainQueue:
    queue: Deque[MergeEvent] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_flush: float = field(default_factory=time.time)


class MergeCoordinator:
    """Coalesce merge events per domain with deterministic ordering."""

    def __init__(
        self,
        curator_service,
        batch_size: int = 8,
        flush_interval: float = 5.0,
    ):
        self.curator_service = curator_service
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.5, flush_interval)
        self._queues: Dict[str, DomainQueue] = defaultdict(DomainQueue)

    def submit(self, domain_id: str, event: MergeEvent):
        queue = self._queues[domain_id]
        should_flush = False
        with queue.lock:
            queue.queue.append(event)
            queue_size = len(queue.queue)
            elapsed = time.time() - queue.last_flush
            should_flush = queue_size >= self.batch_size or elapsed >= self.flush_interval

        if should_flush:
            return self.flush(domain_id)
        return None

    def flush(self, domain_id: str):
        queue = self._queues[domain_id]
        with queue.lock:
            if not queue.queue:
                return None
            events = list(queue.queue)
            queue.queue.clear()
            queue.last_flush = time.time()

        target_stages = {event.target_stage for event in events}
        if len(target_stages) > 1:
            logger.warning(
                "merge_coordinator_mixed_stages",
                domain_id=domain_id,
                stages=list(target_stages),
            )
        target_stage = target_stages.pop() if target_stages else PlaybookStage.SHADOW

        task_insights = [
            {
                "task_id": event.task_id,
                "domain_id": domain_id,
                "insights": event.insights,
            }
            for event in events
        ]

        logger.info(
            "merge_coordinator_flush",
            domain_id=domain_id,
            batch_size=len(task_insights),
            target_stage=target_stage,
        )

        return self.curator_service.merge_batch(
            domain_id=domain_id,
            task_insights=task_insights,
            target_stage=target_stage,
        )

    def flush_all(self):
        results = []
        for domain_id in list(self._queues.keys()):
            result = self.flush(domain_id)
            if result is not None:
                results.append(result)
        return results

