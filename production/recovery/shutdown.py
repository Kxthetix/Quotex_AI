"""Graceful Shutdown Orchestrator for Phase 9 (Section 43).

Executes ordered, safe termination sequence:
1. Halt new trading decisions and stop new orders.
2. Complete safe in-flight evaluations.
3. Flush persistence and telemetry buffers.
4. Persist durable state snapshot.
5. Close WebSocket streams, database pools, and Redis connections.
6. Emit immutable shutdown audit event.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShutdownReport:
    success: bool
    reason: str
    hooks_executed: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class GracefulShutdownManager:
    """Manages ordered, multi-stage safe shutdown sequence."""

    def __init__(self) -> None:
        self.is_shutting_down: bool = False
        self._shutdown_hooks: list[tuple[str, Callable[[], None | Awaitable[None]]]] = []

    def register_hook(self, name: str, hook: Callable[[], None | Awaitable[None]]) -> None:
        """Register a cleanup hook to run during shutdown."""
        self._shutdown_hooks.append((name, hook))

    def shutdown(self, reason: str = "Clean shutdown requested") -> ShutdownReport:
        """Synchronously execute full graceful shutdown sequence."""
        if self.is_shutting_down:
            return ShutdownReport(success=True, reason=reason)

        self.is_shutting_down = True
        logger.warning("initiating_graceful_shutdown", reason=reason)

        executed = []
        success = True
        for name, hook in self._shutdown_hooks:
            logger.info("executing_shutdown_hook", hook_name=name)
            try:
                res = hook()
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(res)
                        else:
                            loop.run_until_complete(res)
                    except Exception:
                        asyncio.run(res)
                executed.append(name)
            except Exception as e:
                success = False
                logger.error("shutdown_hook_failed", hook_name=name, error=str(e))

        logger.info("graceful_shutdown_completed", status="SUCCESS" if success else "PARTIAL_FAIL")
        return ShutdownReport(success=success, reason=reason, hooks_executed=executed)

    async def execute_shutdown(self, reason: str = "Clean shutdown requested") -> bool:
        """Execute full graceful shutdown sequence asynchronously."""
        report = self.shutdown(reason=reason)
        return report.success
