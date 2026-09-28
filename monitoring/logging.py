"""Structured logging configuration for Trading Intelligence System.

Configures structlog to output either colored development logs or machine-readable JSON logs.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to all log events."""
    event_dict.setdefault("app", "trading-system")
    return event_dict


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Configure system-wide logging using structlog and standard logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Common processors for both JSON and console modes
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format.lower() == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also configure standard library root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog bound logger."""
    return structlog.get_logger(name)
