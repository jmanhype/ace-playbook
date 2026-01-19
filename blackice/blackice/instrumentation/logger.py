"""Structured logging configuration for BLACKICE 3.0.

Configures structlog for JSON logging with correlation IDs
and context propagation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def add_correlation_id(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add correlation ID from context if available."""
    from blackice.instrumentation.context import get_correlation_id

    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_run_context(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add run context (run_id, task_id) if available."""
    from blackice.instrumentation.context import get_run_context

    ctx = get_run_context()
    if ctx.get("run_id"):
        event_dict["run_id"] = str(ctx["run_id"])
    if ctx.get("task_id"):
        event_dict["task_id"] = str(ctx["task_id"])
    return event_dict


def redact_secrets(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Redact potential secrets from log output."""
    import re

    # Patterns that might indicate secrets
    secret_patterns = [
        (r"(api[_-]?key)[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?", r"\1=[REDACTED]"),
        (r"(token)[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?", r"\1=[REDACTED]"),
        (r"(password)[=:]\s*['\"]?([^\s'\"]{4,})['\"]?", r"\1=[REDACTED]"),
        (r"(secret)[=:]\s*['\"]?([a-zA-Z0-9_-]{10,})['\"]?", r"\1=[REDACTED]"),
        (r"sk-[a-zA-Z0-9]{32,}", "[REDACTED_API_KEY]"),
    ]

    def redact_value(value: Any) -> Any:
        if isinstance(value, str):
            for pattern, replacement in secret_patterns:
                value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        elif isinstance(value, dict):
            return {k: redact_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [redact_value(v) for v in value]
        return value

    return {k: redact_value(v) for k, v in event_dict.items()}


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    add_timestamp: bool = True,
    redact_secrets_enabled: bool = True,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON formatting (vs human-readable)
        add_timestamp: Add ISO timestamp to each log
        redact_secrets_enabled: Enable secret redaction
    """
    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso"))

    # Add BLACKICE-specific processors
    shared_processors.append(add_correlation_id)
    shared_processors.append(add_run_context)

    if redact_secrets_enabled:
        shared_processors.append(redact_secrets)

    # Configure for JSON or console output
    if json_output:
        shared_processors.append(structlog.processors.format_exc_info)
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Set library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context.

    Example:
        with LogContext(run_id=run.id, task="planning"):
            logger.info("Starting task")  # Includes run_id and task
    """

    def __init__(self, **kwargs: Any) -> None:
        self._context = kwargs
        self._token: Any = None

    def __enter__(self) -> LogContext:
        self._token = structlog.contextvars.bind_contextvars(**self._context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self._context.keys())

    async def __aenter__(self) -> LogContext:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)
