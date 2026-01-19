"""OpenTelemetry tracing configuration for BLACKICE 3.0.

Provides distributed tracing with span creation and context propagation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, Span, Tracer

from blackice.instrumentation.context import get_correlation_id, get_run_context


def configure_tracing(
    service_name: str = "blackice",
    otlp_endpoint: str | None = None,
    console_export: bool = False,
    sample_rate: float = 1.0,
) -> Tracer:
    """Configure OpenTelemetry tracing.

    Args:
        service_name: Name of the service for traces
        otlp_endpoint: OTLP exporter endpoint (e.g., "localhost:4317")
        console_export: Also export to console (for debugging)
        sample_rate: Sampling rate (0.0-1.0)

    Returns:
        Configured Tracer instance
    """
    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "3.0.0",
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter for debugging
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)


def get_tracer(name: str = "blackice") -> Tracer:
    """Get a tracer instance.

    Args:
        name: Tracer name

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


@contextmanager
def create_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[Span, None, None]:
    """Create a new span for tracing.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)

    Yields:
        The created span
    """
    tracer = get_tracer()

    # Build attributes with context
    span_attributes: dict[str, Any] = attributes or {}

    # Add correlation ID
    correlation_id = get_correlation_id()
    if correlation_id:
        span_attributes["correlation_id"] = correlation_id

    # Add run context
    ctx = get_run_context()
    if ctx.get("run_id"):
        span_attributes["run_id"] = str(ctx["run_id"])
    if ctx.get("task_id"):
        span_attributes["task_id"] = str(ctx["task_id"])
    if ctx.get("agent_id"):
        span_attributes["agent_id"] = ctx["agent_id"]

    with tracer.start_as_current_span(name, kind=kind, attributes=span_attributes) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


class SpanContext:
    """Context manager for creating spans with automatic error handling.

    Example:
        async with SpanContext("process_task", task_id=task.id) as span:
            result = await process()
            span.set_attribute("result_size", len(result))
    """

    def __init__(
        self,
        name: str,
        *,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        **attributes: Any,
    ) -> None:
        self.name = name
        self.kind = kind
        self.attributes = attributes
        self._span: Span | None = None
        self._token: Any = None

    def __enter__(self) -> Span:
        tracer = get_tracer()

        # Build attributes
        span_attributes = dict(self.attributes)
        correlation_id = get_correlation_id()
        if correlation_id:
            span_attributes["correlation_id"] = correlation_id

        ctx = get_run_context()
        if ctx.get("run_id"):
            span_attributes["run_id"] = str(ctx["run_id"])
        if ctx.get("task_id"):
            span_attributes["task_id"] = str(ctx["task_id"])

        self._span = tracer.start_span(self.name, kind=self.kind, attributes=span_attributes)
        self._token = trace.use_span(self._span, end_on_exit=False)
        self._token.__enter__()
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span:
            if exc_val:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self._span.record_exception(exc_val)
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._token.__exit__(exc_type, exc_val, exc_tb)
            self._span.end()

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def trace_function(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        attributes: Static span attributes

    Example:
        @trace_function()
        async def process_request(request):
            ...
    """
    import functools
    from typing import Callable, TypeVar

    F = TypeVar("F", bound=Callable[..., Any])

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with create_span(span_name, attributes=attributes):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with create_span(span_name, attributes=attributes):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
