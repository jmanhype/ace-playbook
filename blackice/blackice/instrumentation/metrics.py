"""Prometheus metrics for BLACKICE 3.0.

Provides metrics collection for monitoring run performance,
token usage, and system health.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

# Note: In production, you'd use prometheus_client
# Here we provide a lightweight in-memory implementation
# that can be exported to Prometheus format


@dataclass
class Counter:
    """A counter metric that only goes up."""

    name: str
    description: str
    labels: tuple[str, ...] = field(default_factory=tuple)
    _values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment the counter."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) + value

    def get(self, **label_values: str) -> float:
        """Get counter value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)


@dataclass
class Gauge:
    """A gauge metric that can go up and down."""

    name: str
    description: str
    labels: tuple[str, ...] = field(default_factory=tuple)
    _values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def set(self, value: float, **label_values: str) -> None:
        """Set gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, **label_values: str) -> None:
        """Decrement gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) - value

    def get(self, **label_values: str) -> float:
        """Get gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)


@dataclass
class Histogram:
    """A histogram metric for measuring distributions."""

    name: str
    description: str
    labels: tuple[str, ...] = field(default_factory=tuple)
    buckets: tuple[float, ...] = field(
        default_factory=lambda: (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
    )
    _counts: dict[tuple[str, ...], list[int]] = field(default_factory=dict)
    _sums: dict[tuple[str, ...], float] = field(default_factory=dict)
    _totals: dict[tuple[str, ...], int] = field(default_factory=dict)

    def observe(self, value: float, **label_values: str) -> None:
        """Observe a value."""
        key = tuple(label_values.get(l, "") for l in self.labels)

        if key not in self._counts:
            self._counts[key] = [0] * len(self.buckets)
            self._sums[key] = 0.0
            self._totals[key] = 0

        # Update bucket counts
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self._counts[key][i] += 1

        self._sums[key] += value
        self._totals[key] += 1

    @contextmanager
    def time(self, **label_values: str) -> Generator[None, None, None]:
        """Context manager to time an operation."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.observe(time.monotonic() - start, **label_values)


class MetricsRegistry:
    """Registry for all BLACKICE metrics."""

    def __init__(self) -> None:
        # Run metrics
        self.runs_total = Counter(
            "blackice_runs_total",
            "Total number of runs started",
            labels=("edition", "status"),
        )

        self.runs_active = Gauge(
            "blackice_runs_active",
            "Number of currently active runs",
            labels=("edition",),
        )

        self.run_duration = Histogram(
            "blackice_run_duration_seconds",
            "Run duration in seconds",
            labels=("edition", "status"),
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
        )

        # Task metrics
        self.tasks_total = Counter(
            "blackice_tasks_total",
            "Total number of tasks executed",
            labels=("status",),
        )

        self.task_duration = Histogram(
            "blackice_task_duration_seconds",
            "Task duration in seconds",
            labels=("status",),
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
        )

        self.task_retries = Counter(
            "blackice_task_retries_total",
            "Total number of task retries",
        )

        # Model metrics
        self.model_calls_total = Counter(
            "blackice_model_calls_total",
            "Total number of model API calls",
            labels=("provider", "model", "status"),
        )

        self.model_tokens_total = Counter(
            "blackice_model_tokens_total",
            "Total tokens used",
            labels=("provider", "model", "type"),  # type: prompt/completion
        )

        self.model_cost_total = Counter(
            "blackice_model_cost_usd_total",
            "Total cost in USD",
            labels=("provider", "model"),
        )

        self.model_latency = Histogram(
            "blackice_model_latency_seconds",
            "Model call latency in seconds",
            labels=("provider", "model"),
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )

        # Command execution metrics
        self.commands_total = Counter(
            "blackice_commands_total",
            "Total commands executed",
            labels=("status",),  # success/failed/blocked
        )

        self.commands_blocked = Counter(
            "blackice_commands_blocked_total",
            "Commands blocked by safety pipeline",
            labels=("policy",),
        )

        # Memory metrics
        self.memory_operations = Counter(
            "blackice_memory_operations_total",
            "Memory provider operations",
            labels=("operation",),  # put/get/search
        )

        # Health metrics
        self.provider_health = Gauge(
            "blackice_provider_health",
            "Provider health status (1=healthy, 0=unhealthy)",
            labels=("provider", "type"),
        )

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines: list[str] = []

        def export_counter(counter: Counter) -> None:
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for labels, value in counter._values.items():
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(counter.labels, labels) if v
                )
                if label_str:
                    lines.append(f"{counter.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{counter.name} {value}")

        def export_gauge(gauge: Gauge) -> None:
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for labels, value in gauge._values.items():
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(gauge.labels, labels) if v
                )
                if label_str:
                    lines.append(f"{gauge.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{gauge.name} {value}")

        def export_histogram(hist: Histogram) -> None:
            lines.append(f"# HELP {hist.name} {hist.description}")
            lines.append(f"# TYPE {hist.name} histogram")
            for labels, counts in hist._counts.items():
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(hist.labels, labels) if v
                )
                base_labels = f"{{{label_str}}}" if label_str else ""

                # Bucket counts
                cumulative = 0
                for bucket, count in zip(hist.buckets, counts):
                    cumulative += count
                    bucket_label = f'{label_str},le="{bucket}"' if label_str else f'le="{bucket}"'
                    lines.append(f"{hist.name}_bucket{{{bucket_label}}} {cumulative}")

                # +Inf bucket
                total = hist._totals.get(labels, 0)
                inf_label = f'{label_str},le="+Inf"' if label_str else 'le="+Inf"'
                lines.append(f"{hist.name}_bucket{{{inf_label}}} {total}")

                # Sum and count
                sum_val = hist._sums.get(labels, 0)
                lines.append(f"{hist.name}_sum{base_labels} {sum_val}")
                lines.append(f"{hist.name}_count{base_labels} {total}")

        # Export all metrics
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, Counter):
                export_counter(obj)
            elif isinstance(obj, Gauge):
                export_gauge(obj)
            elif isinstance(obj, Histogram):
                export_histogram(obj)

        return "\n".join(lines)


# Global metrics registry
_metrics: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics registry (for testing)."""
    global _metrics
    _metrics = None
