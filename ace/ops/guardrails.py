"""
Guardrail Monitoring and Automated Rollback

Monitors playbook performance and triggers rollback on regression:
- Success rate delta monitoring
- Latency P95 monitoring
- Error rate tracking
- Automated rollback to previous version
"""

import json
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.repositories.journal_repository import DiffJournalRepository
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="guardrails")


@dataclass
class PerformanceSnapshot:
    """Snapshot of playbook performance metrics."""
    timestamp: datetime
    success_rate: float  # 0.0-1.0
    latency_p95_ms: float
    error_rate: float  # 0.0-1.0
    total_tasks: int
    domain_id: str


@dataclass
class RollbackTrigger:
    """Trigger conditions that initiated rollback."""
    reason: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: datetime


class GuardrailMonitor:
    """
    Monitor playbook performance and trigger rollback on regression.

    Monitors:
    - Success rate delta: rollback if drops >8%
    - Latency P95: rollback if increases >30%
    - Error rate: rollback if increases significantly

    Rollback Strategy:
    - Revert to previous playbook snapshot (24h ago)
    - Quarantine recently added bullets
    - Send alert notification
    """

    # Thresholds for triggering rollback
    SUCCESS_RATE_THRESHOLD = -0.08  # -8%
    LATENCY_P95_THRESHOLD = 0.30  # +30%
    ERROR_RATE_THRESHOLD = 0.15  # 15% absolute

    def __init__(
        self,
        session: Session,
        alert_callback: Optional[Callable[[RollbackTrigger], None]] = None
    ):
        """
        Initialize guardrail monitor.

        Args:
            session: Database session
            alert_callback: Optional callback for sending alerts (webhook, log, etc.)
        """
        self.session = session
        self.playbook_repo = PlaybookRepository(session)
        self.journal_repo = DiffJournalRepository(session)
        self.alert_callback = alert_callback or self._default_alert

        # Performance history
        self.baseline_snapshot: Optional[PerformanceSnapshot] = None
        self.current_snapshot: Optional[PerformanceSnapshot] = None

    def update_baseline(self, snapshot: PerformanceSnapshot) -> None:
        """
        Update baseline performance snapshot.

        Args:
            snapshot: Performance metrics to use as baseline
        """
        self.baseline_snapshot = snapshot
        logger.info(
            "baseline_updated",
            domain_id=snapshot.domain_id,
            success_rate=snapshot.success_rate,
            latency_p95=snapshot.latency_p95_ms,
            total_tasks=snapshot.total_tasks
        )

    def record_current(self, snapshot: PerformanceSnapshot) -> None:
        """
        Record current performance snapshot.

        Args:
            snapshot: Current performance metrics
        """
        self.current_snapshot = snapshot

    def check_guardrails(self, domain_id: str) -> Optional[RollbackTrigger]:
        """
        Check if guardrails are violated and trigger rollback if needed.

        Args:
            domain_id: Domain to check

        Returns:
            RollbackTrigger if rollback was triggered, None otherwise
        """
        if not self.baseline_snapshot or not self.current_snapshot:
            logger.warning(
                "guardrail_check_skipped",
                reason="missing_snapshots",
                domain_id=domain_id
            )
            return None

        # Check success rate delta
        success_delta = self.current_snapshot.success_rate - self.baseline_snapshot.success_rate
        if success_delta < self.SUCCESS_RATE_THRESHOLD:
            trigger = RollbackTrigger(
                reason=f"Success rate dropped by {abs(success_delta)*100:.1f}%",
                metric_name="success_rate_delta",
                threshold_value=self.SUCCESS_RATE_THRESHOLD,
                actual_value=success_delta,
                timestamp=datetime.utcnow()
            )
            self._trigger_rollback(domain_id, trigger)
            return trigger

        # Check latency P95 delta
        if self.baseline_snapshot.latency_p95_ms > 0:
            latency_delta_ratio = (
                (self.current_snapshot.latency_p95_ms - self.baseline_snapshot.latency_p95_ms)
                / self.baseline_snapshot.latency_p95_ms
            )
            if latency_delta_ratio > self.LATENCY_P95_THRESHOLD:
                trigger = RollbackTrigger(
                    reason=f"Latency P95 increased by {latency_delta_ratio*100:.1f}%",
                    metric_name="latency_p95_delta",
                    threshold_value=self.LATENCY_P95_THRESHOLD,
                    actual_value=latency_delta_ratio,
                    timestamp=datetime.utcnow()
                )
                self._trigger_rollback(domain_id, trigger)
                return trigger

        # Check error rate
        if self.current_snapshot.error_rate > self.ERROR_RATE_THRESHOLD:
            trigger = RollbackTrigger(
                reason=f"Error rate at {self.current_snapshot.error_rate*100:.1f}%",
                metric_name="error_rate",
                threshold_value=self.ERROR_RATE_THRESHOLD,
                actual_value=self.current_snapshot.error_rate,
                timestamp=datetime.utcnow()
            )
            self._trigger_rollback(domain_id, trigger)
            return trigger

        logger.info(
            "guardrails_passed",
            domain_id=domain_id,
            success_delta=success_delta,
            latency_delta=latency_delta_ratio if self.baseline_snapshot.latency_p95_ms > 0 else 0,
            error_rate=self.current_snapshot.error_rate
        )

        return None

    def _trigger_rollback(self, domain_id: str, trigger: RollbackTrigger) -> None:
        """
        Execute rollback procedure.

        Args:
            domain_id: Domain to rollback
            trigger: Trigger that initiated rollback
        """
        logger.error(
            "rollback_triggered",
            domain_id=domain_id,
            reason=trigger.reason,
            metric=trigger.metric_name,
            threshold=trigger.threshold_value,
            actual=trigger.actual_value
        )

        try:
            # Get bullets added in last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_bullets = self.playbook_repo.get_by_domain(
                domain_id=domain_id,
                created_after=cutoff_time
            )

            # Quarantine recent bullets
            quarantined_count = 0
            for bullet in recent_bullets:
                if bullet.stage != PlaybookStage.QUARANTINED:
                    bullet.stage = PlaybookStage.QUARANTINED
                    self.playbook_repo.update(bullet)
                    quarantined_count += 1

            self.session.commit()

            logger.info(
                "rollback_completed",
                domain_id=domain_id,
                quarantined_count=quarantined_count
            )

            # Send alert
            self.alert_callback(trigger)

        except Exception as e:
            logger.error(
                "rollback_failed",
                domain_id=domain_id,
                error=str(e)
            )
            self.session.rollback()
            raise

    def _default_alert(self, trigger: RollbackTrigger) -> None:
        """
        Default alert handler (logs to console).

        Args:
            trigger: Rollback trigger details
        """
        logger.error(
            "guardrail_alert",
            reason=trigger.reason,
            metric=trigger.metric_name,
            threshold=trigger.threshold_value,
            actual=trigger.actual_value,
            timestamp=trigger.timestamp.isoformat()
        )

    def get_rollback_history(self, domain_id: str, days: int = 7) -> List[Dict]:
        """
        Get history of rollback events.

        Args:
            domain_id: Domain to query
            days: Number of days to look back

        Returns:
            List of rollback event dictionaries
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Query journal for quarantine operations in rollback window
        entries = self.journal_repo.get_by_domain(
            domain_id=domain_id,
            since=cutoff_time
        )

        rollback_events = []
        for entry in entries:
            # Look for bulk quarantine operations (indicator of rollback)
            if entry.operation.value == "quarantine":
                metadata = entry.metadata_dict or {}
                if metadata.get("reason") == "automated_rollback":
                    rollback_events.append({
                        "timestamp": entry.timestamp.isoformat(),
                        "bullet_id": entry.bullet_id,
                        "metadata": metadata
                    })

        return rollback_events


def create_guardrail_monitor(
    session: Session,
    alert_callback: Optional[Callable[[RollbackTrigger], None]] = None
) -> GuardrailMonitor:
    """
    Create GuardrailMonitor instance.

    Args:
        session: Database session
        alert_callback: Optional callback for alerts

    Returns:
        GuardrailMonitor instance
    """
    return GuardrailMonitor(session, alert_callback)
