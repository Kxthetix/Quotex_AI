"""Retry Policy with Exponential Backoff and Jitter (Section 9).

Prevents thundering herd problems, infinite tight loops, and cascading failures
through configurable exponential backoff with full or equal jitter.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from monitoring.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configurable retry and backoff parameters (Section 9)."""
    max_attempts: int = 4
    initial_delay: float = 0.5
    max_delay: float = 10.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)


class RetryPolicy:
    """Executes callables with exponential backoff, jitter, and safety limits."""

    def __init__(
        self,
        config: RetryConfig | None = None,
        max_attempts: int | None = None,
        initial_delay: float | None = None,
        max_delay: float | None = None,
        backoff_factor: float | None = None,
        backoff_multiplier: float | None = None,
        jitter: bool | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            mult = backoff_factor if backoff_factor is not None else (backoff_multiplier or 2.0)
            self.config = RetryConfig(
                max_attempts=max_attempts if max_attempts is not None else 4,
                initial_delay=initial_delay if initial_delay is not None else 0.5,
                max_delay=max_delay if max_delay is not None else 10.0,
                backoff_multiplier=mult,
                jitter=jitter if jitter is not None else True,
                retryable_exceptions=retryable_exceptions or (Exception,),
            )

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Alias for execute_sync."""
        return self.execute_sync(func, *args, **kwargs)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and randomized jitter."""
        delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Full jitter: random between 0 and delay
            delay = random.uniform(0.5 * delay, delay)

        return max(0.01, delay)

    def execute_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute synchronous function with retry logic."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                if attempt >= self.config.max_attempts:
                    logger.error("retry_exhausted", attempts=attempt, error=str(e))
                    raise
                delay = self.calculate_delay(attempt)
                logger.warning(
                    "retry_attempt",
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    delay_s=round(delay, 2),
                    error=str(e),
                )
                time.sleep(delay)

    async def execute_async(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute asynchronous coroutine with retry logic."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                if attempt >= self.config.max_attempts:
                    logger.error("async_retry_exhausted", attempts=attempt, error=str(e))
                    raise
                delay = self.calculate_delay(attempt)
                logger.warning(
                    "async_retry_attempt",
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    delay_s=round(delay, 2),
                    error=str(e),
                )
                await asyncio.sleep(delay)
