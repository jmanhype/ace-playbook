"""
Logging Configuration for ACE Framework

Structured JSON logging with contextual fields for observability.
"""

import logging
import sys
import structlog
from typing import Any


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structured logging for ACE framework.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" for production, "console" for development)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with contextual fields.

    Args:
        name: Logger name (typically __name__)
        **context: Additional context fields to bind to all log entries

    Returns:
        Bound logger with context

    Example:
        logger = get_logger(__name__, component="curator", domain_id="customer-acme")
        logger.info("deduplication_complete", new_bullets=5, duplicates=2)
        # Output: {"event": "deduplication_complete", "component": "curator",
        #          "domain_id": "customer-acme", "new_bullets": 5, "duplicates": 2}
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger
