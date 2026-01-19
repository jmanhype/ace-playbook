"""Instrumentation for BLACKICE 3.0.

Provides structured logging, tracing, metrics, and context management.
"""

from blackice.instrumentation.context import (
    InstrumentationContext,
    clear_context,
    get_correlation_id,
    get_run_context,
    set_agent_id,
    set_correlation_id,
    set_run_id,
    set_task_id,
)
from blackice.instrumentation.logger import (
    LogContext,
    configure_logging,
    get_logger,
)
from blackice.instrumentation.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics,
    reset_metrics,
)
from blackice.instrumentation.tracing import (
    SpanContext,
    configure_tracing,
    create_span,
    get_tracer,
    trace_function,
)

__all__ = [
    # Context
    "get_correlation_id",
    "set_correlation_id",
    "get_run_context",
    "set_run_id",
    "set_task_id",
    "set_agent_id",
    "clear_context",
    "InstrumentationContext",
    # Logger
    "configure_logging",
    "get_logger",
    "LogContext",
    # Tracing
    "configure_tracing",
    "get_tracer",
    "create_span",
    "SpanContext",
    "trace_function",
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_metrics",
    "reset_metrics",
]
