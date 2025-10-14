"""
Observability Metrics for ACE Playbook

Prometheus-style metrics for monitoring playbook operations, performance,
and learning progress.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Thread-safe metrics collector for ACE operations.

    Tracks:
    - Playbook size and growth
    - Retrieval performance
    - Reflection operations
    - Stage promotions and quarantine events
    """

    def __init__(self):
        """Initialize metrics collector with thread-safe counters."""
        self._lock = Lock()

        # Gauge metrics (current value)
        self._playbook_bullet_count: Dict[str, int] = {}  # by domain_id

        # Counter metrics (cumulative)
        self._reflection_count: int = 0
        self._promotion_events_total: int = 0
        self._quarantine_events_total: int = 0
        self._bullets_added_total: int = 0
        self._bullets_incremented_total: int = 0

        # Histogram metrics (latency tracking)
        self._retrieval_latencies: list[float] = []  # ms
        self._reflection_latencies: list[float] = []  # ms
        self._curator_latencies: list[float] = []  # ms

    def record_bullet_count(self, domain_id: str, count: int) -> None:
        """
        Record current playbook bullet count for a domain.

        Args:
            domain_id: Domain namespace
            count: Current bullet count
        """
        with self._lock:
            self._playbook_bullet_count[domain_id] = count

    def increment_bullets_added(self, count: int = 1) -> None:
        """Increment counter for new bullets added."""
        with self._lock:
            self._bullets_added_total += count

    def increment_bullets_incremented(self, count: int = 1) -> None:
        """Increment counter for existing bullets updated."""
        with self._lock:
            self._bullets_incremented_total += count

    def record_retrieval_latency(self, latency_ms: float) -> None:
        """
        Record playbook retrieval latency.

        Args:
            latency_ms: Retrieval time in milliseconds
        """
        with self._lock:
            self._retrieval_latencies.append(latency_ms)
            # Keep last 1000 measurements
            if len(self._retrieval_latencies) > 1000:
                self._retrieval_latencies.pop(0)

    def increment_reflection_count(self) -> None:
        """Increment counter for reflection operations."""
        with self._lock:
            self._reflection_count += 1

    def record_reflection_latency(self, latency_ms: float) -> None:
        """
        Record reflection operation latency.

        Args:
            latency_ms: Reflection time in milliseconds
        """
        with self._lock:
            self._reflection_latencies.append(latency_ms)
            if len(self._reflection_latencies) > 1000:
                self._reflection_latencies.pop(0)

    def record_curator_latency(self, latency_ms: float) -> None:
        """
        Record curator merge operation latency.

        Args:
            latency_ms: Curator time in milliseconds
        """
        with self._lock:
            self._curator_latencies.append(latency_ms)
            if len(self._curator_latencies) > 1000:
                self._curator_latencies.pop(0)

    def increment_promotion_events(self) -> None:
        """Increment counter for bullet promotions."""
        with self._lock:
            self._promotion_events_total += 1

    def increment_quarantine_events(self) -> None:
        """Increment counter for bullet quarantines."""
        with self._lock:
            self._quarantine_events_total += 1

    def get_metrics(self) -> Dict[str, any]:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary of all current metrics
        """
        with self._lock:
            return {
                "playbook_bullet_count": dict(self._playbook_bullet_count),
                "reflection_count": self._reflection_count,
                "promotion_events_total": self._promotion_events_total,
                "quarantine_events_total": self._quarantine_events_total,
                "bullets_added_total": self._bullets_added_total,
                "bullets_incremented_total": self._bullets_incremented_total,
                "retrieval_latency_ms": self._compute_latency_stats(self._retrieval_latencies),
                "reflection_latency_ms": self._compute_latency_stats(self._reflection_latencies),
                "curator_latency_ms": self._compute_latency_stats(self._curator_latencies),
            }

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self.get_metrics()
        lines = []

        # Gauge: playbook_bullet_count by domain
        lines.append("# HELP playbook_bullet_count Current number of bullets in playbook by domain")
        lines.append("# TYPE playbook_bullet_count gauge")
        for domain_id, count in metrics["playbook_bullet_count"].items():
            lines.append(f'playbook_bullet_count{{domain_id="{domain_id}"}} {count}')

        # Counter: reflection_count
        lines.append("# HELP reflection_count Total number of reflection operations")
        lines.append("# TYPE reflection_count counter")
        lines.append(f"reflection_count {metrics['reflection_count']}")

        # Counter: promotion_events_total
        lines.append("# HELP promotion_events_total Total number of bullet promotions")
        lines.append("# TYPE promotion_events_total counter")
        lines.append(f"promotion_events_total {metrics['promotion_events_total']}")

        # Counter: quarantine_events_total
        lines.append("# HELP quarantine_events_total Total number of bullet quarantines")
        lines.append("# TYPE quarantine_events_total counter")
        lines.append(f"quarantine_events_total {metrics['quarantine_events_total']}")

        # Counter: bullets_added_total
        lines.append("# HELP bullets_added_total Total number of new bullets added")
        lines.append("# TYPE bullets_added_total counter")
        lines.append(f"bullets_added_total {metrics['bullets_added_total']}")

        # Counter: bullets_incremented_total
        lines.append("# HELP bullets_incremented_total Total number of existing bullets incremented")
        lines.append("# TYPE bullets_incremented_total counter")
        lines.append(f"bullets_incremented_total {metrics['bullets_incremented_total']}")

        # Histogram: retrieval_latency_ms
        latency = metrics["retrieval_latency_ms"]
        if latency:
            lines.append("# HELP retrieval_latency_ms Playbook retrieval latency in milliseconds")
            lines.append("# TYPE retrieval_latency_ms summary")
            lines.append(f"retrieval_latency_ms{{quantile=\"0.5\"}} {latency.get('p50', 0)}")
            lines.append(f"retrieval_latency_ms{{quantile=\"0.95\"}} {latency.get('p95', 0)}")
            lines.append(f"retrieval_latency_ms{{quantile=\"0.99\"}} {latency.get('p99', 0)}")
            lines.append(f"retrieval_latency_ms_sum {latency.get('sum', 0)}")
            lines.append(f"retrieval_latency_ms_count {latency.get('count', 0)}")

        # Histogram: reflection_latency_ms
        latency = metrics["reflection_latency_ms"]
        if latency:
            lines.append("# HELP reflection_latency_ms Reflection operation latency in milliseconds")
            lines.append("# TYPE reflection_latency_ms summary")
            lines.append(f"reflection_latency_ms{{quantile=\"0.5\"}} {latency.get('p50', 0)}")
            lines.append(f"reflection_latency_ms{{quantile=\"0.95\"}} {latency.get('p95', 0)}")
            lines.append(f"reflection_latency_ms{{quantile=\"0.99\"}} {latency.get('p99', 0)}")
            lines.append(f"reflection_latency_ms_sum {latency.get('sum', 0)}")
            lines.append(f"reflection_latency_ms_count {latency.get('count', 0)}")

        # Histogram: curator_latency_ms
        latency = metrics["curator_latency_ms"]
        if latency:
            lines.append("# HELP curator_latency_ms Curator merge operation latency in milliseconds")
            lines.append("# TYPE curator_latency_ms summary")
            lines.append(f"curator_latency_ms{{quantile=\"0.5\"}} {latency.get('p50', 0)}")
            lines.append(f"curator_latency_ms{{quantile=\"0.95\"}} {latency.get('p95', 0)}")
            lines.append(f"curator_latency_ms{{quantile=\"0.99\"}} {latency.get('p99', 0)}")
            lines.append(f"curator_latency_ms_sum {latency.get('sum', 0)}")
            lines.append(f"curator_latency_ms_count {latency.get('count', 0)}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _compute_latency_stats(latencies: list[float]) -> Optional[Dict[str, float]]:
        """Compute latency percentiles."""
        if not latencies:
            return None

        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(count * 0.5)],
            "p95": sorted_latencies[int(count * 0.95)],
            "p99": sorted_latencies[int(count * 0.99)],
            "sum": sum(sorted_latencies),
            "count": count,
            "mean": sum(sorted_latencies) / count,
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        Singleton MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class LatencyTimer:
    """
    Context manager for timing operations and recording latency.

    Usage:
        with LatencyTimer() as timer:
            # perform operation
            pass
        metrics.record_retrieval_latency(timer.elapsed_ms())
    """

    def __init__(self):
        """Initialize timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "LatencyTimer":
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer."""
        self.end_time = time.perf_counter()
        return False

    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
