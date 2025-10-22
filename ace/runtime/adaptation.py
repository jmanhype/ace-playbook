"""Runtime adaptation helpers for test-time learning."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple

from ace.curator.merge_coordinator import MergeCoordinator, MergeEvent
from ace.models.playbook import PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="runtime_adapter")


class RuntimeAdapter:
    """In-memory cache for in-flight insights prior to durable merge."""

    def __init__(
        self,
        domain_id: str,
        merge_coordinator: MergeCoordinator,
        ttl_seconds: float = 300.0,
    ):
        self.domain_id = domain_id
        self.merge_coordinator = merge_coordinator
        self.ttl = max(30.0, ttl_seconds)
        self._lock = threading.Lock()
        self._entries: List[Tuple[float, Dict]] = []

    def ingest(self, task_id: str, insight: Dict) -> None:
        """Ingest a single insight and enqueue for merge."""

        timestamp = time.time()
        normalized = {
            "id": insight.get("id", f"runtime-{int(timestamp)}"),
            "content": insight.get("content", ""),
            "stage": insight.get("stage", "shadow"),
            "helpful_count": insight.get("helpful_count", 0),
            "harmful_count": insight.get("harmful_count", 0),
        }

        with self._lock:
            self._entries.append((timestamp, normalized))

        merge_event = MergeEvent(
            task_id=task_id,
            insights=[
                {
                    "content": insight.get("content", ""),
                    "section": insight.get("section", "Helpful"),
                    "tags": insight.get("tags", []),
                    "metadata": {"source_task_id": task_id},
                }
            ],
            target_stage=PlaybookStage.SHADOW,
        )

        result = self.merge_coordinator.submit(self.domain_id, merge_event)
        if result:
            logger.info(
                "runtime_adapter_merged",
                domain_id=self.domain_id,
                new_bullets=result.new_bullets_added,
                increments=result.existing_bullets_incremented,
            )

    def get_hot_entries(self) -> List[Dict]:
        """Return TTL-filtered context entries for immediate use."""

        now = time.time()
        with self._lock:
            self._entries = [
                (ts, entry)
                for ts, entry in self._entries
                if now - ts <= self.ttl
            ]
            return [entry for _, entry in self._entries]
