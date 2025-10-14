"""
Distributed tracing with OpenTelemetry for ACE Playbook (T076).

Provides automatic instrumentation for:
- LLM API calls (Generator, Reflector)
- Database operations (SQLAlchemy)
- FAISS similarity searches
- HTTP requests

Usage:
    from ace.utils.tracing import setup_tracing, trace_operation

    # Initialize tracing (call once at startup)
    setup_tracing(service_name="ace-playbook", environment="production")

    # Automatic tracing via decorator
    @trace_operation("custom_operation")
    def my_function():
        pass

    # Manual span creation
    from ace.utils.tracing import get_tracer

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("span_name"):
        # ...

Configuration:
    Set environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4318)
    - OTEL_SERVICE_NAME: Service name (default: ace-playbook)
    - OTEL_ENVIRONMENT: Environment (default: development)
    - OTEL_TRACING_ENABLED: Enable tracing (default: true)
"""

import os
from typing import Optional, Callable, Any
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, DEPLOYMENT_ENVIRONMENT
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="tracing")

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None
_tracing_enabled: bool = True


def setup_tracing(
    service_name: str = "ace-playbook",
    environment: str = "development",
    otlp_endpoint: Optional[str] = None,
    enabled: Optional[bool] = None
) -> None:
    """
    Initialize OpenTelemetry distributed tracing.

    Args:
        service_name: Service name for traces
        environment: Deployment environment (dev, staging, prod)
        otlp_endpoint: OTLP collector endpoint (default: http://localhost:4318)
        enabled: Enable/disable tracing (default: from OTEL_TRACING_ENABLED env var)

    Example:
        setup_tracing(
            service_name="ace-playbook",
            environment="production",
            otlp_endpoint="https://otel-collector.example.com"
        )
    """
    global _tracer_provider, _tracing_enabled

    # Check if tracing is enabled
    if enabled is None:
        enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"

    _tracing_enabled = enabled

    if not _tracing_enabled:
        logger.info("tracing_disabled")
        return

    # Get configuration from environment if not provided
    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)
    environment = os.getenv("OTEL_ENVIRONMENT", environment)
    otlp_endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4318"
    )

    # Create resource with service info
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        DEPLOYMENT_ENVIRONMENT: environment,
        "ace.version": "1.10.0"
    })

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{otlp_endpoint}/v1/traces"
    )

    # Add span processor with batching
    span_processor = BatchSpanProcessor(otlp_exporter)
    _tracer_provider.add_span_processor(span_processor)

    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Auto-instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()

    # Auto-instrument HTTP requests
    RequestsInstrumentor().instrument()

    logger.info(
        "tracing_initialized",
        service_name=service_name,
        environment=environment,
        otlp_endpoint=otlp_endpoint
    )


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Tracer instance

    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("operation"):
            # ...
    """
    if not _tracing_enabled:
        # Return no-op tracer if tracing disabled
        return trace.get_tracer(name)

    return trace.get_tracer(name)


def trace_operation(
    operation_name: str,
    attributes: Optional[dict] = None
) -> Callable:
    """
    Decorator to automatically trace a function execution.

    Args:
        operation_name: Name of the operation (shown in traces)
        attributes: Additional span attributes

    Returns:
        Decorated function

    Example:
        @trace_operation("generate_playbook", {"domain_id": "ml-ops"})
        def generate(task: str) -> str:
            return generator.generate(task)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_enabled:
                # Skip tracing if disabled
                return func(*args, **kwargs)

            tracer = get_tracer(func.__module__)

            with tracer.start_as_current_span(operation_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("operation.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("operation.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def add_span_attributes(**attributes: Any) -> None:
    """
    Add attributes to the current active span.

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        add_span_attributes(
            domain_id="ml-ops",
            task_count=5,
            bullet_count=10
        )
    """
    if not _tracing_enabled:
        return

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict] = None) -> None:
    """
    Add an event to the current active span.

    Args:
        name: Event name
        attributes: Event attributes

    Example:
        add_span_event(
            "bullet_promoted",
            {"bullet_id": "abc123", "stage": "PROD"}
        )
    """
    if not _tracing_enabled:
        return

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes=attributes or {})


def shutdown_tracing() -> None:
    """
    Shutdown tracing and flush all pending spans.

    Call this before application exit to ensure all spans are exported.

    Example:
        import atexit
        atexit.register(shutdown_tracing)
    """
    global _tracer_provider

    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("tracing_shutdown")
