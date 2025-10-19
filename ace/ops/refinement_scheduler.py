"""Periodic maintenance scheduler for ACE refinement tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ace.models.playbook import PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="refinement_scheduler")


@dataclass
class RefinementResult:
    last_run: datetime
    promotions: int
    quarantines: int
    pruned: int
    merged_batches: int


class RefinementScheduler:
    """Coordinate merge flush, promotions, and pruning on a cadence."""

    def __init__(
        self,
        merge_coordinator,
        stage_manager,
        curator_service,
        min_interval_seconds: float = 30.0,
    ):
        self.merge_coordinator = merge_coordinator
        self.stage_manager = stage_manager
        self.curator_service = curator_service
        self.min_interval = timedelta(seconds=max(5.0, min_interval_seconds))
        self._last_run: Optional[datetime] = None

    def should_run(self) -> bool:
        if self._last_run is None:
            return True
        return datetime.utcnow() - self._last_run >= self.min_interval

    def run(self, domain_id: str) -> RefinementResult:
        merged = self.merge_coordinator.flush(domain_id)
        merged_batches = 1 if merged else 0

        pruned_before = self.curator_service.prune_redundant(domain_id)
        promotion_stats = self.stage_manager.check_all_promotions(domain_id)
        pruned_after = self.curator_service.prune_redundant(domain_id)
        pruned = pruned_before + pruned_after

        self._last_run = datetime.utcnow()

        logger.info(
            "refinement_cycle_complete",
            domain_id=domain_id,
            promotions=len(promotion_stats["promoted"]),
            quarantines=len(promotion_stats["quarantined"]),
            pruned=pruned,
            pruned_before=pruned_before,
            pruned_after=pruned_after,
            merged_batches=merged_batches,
        )

        return RefinementResult(
            last_run=self._last_run,
            promotions=len(promotion_stats["promoted"]),
            quarantines=len(promotion_stats["quarantined"]),
            pruned=pruned,
            merged_batches=merged_batches,
        )
