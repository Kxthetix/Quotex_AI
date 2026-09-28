"""System health checks and latency percentile tracking."""

from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from data.ingestion.models import LatencyMetrics
from monitoring.logging import get_logger

logger = get_logger(__name__)


class LatencyTracker:
    """Sliding-window latency tracker measuring p50, p95, and p99 percentiles."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._samples: deque[float] = deque(maxlen=window_size)

    def record_latency_ms(self, latency_ms: float) -> None:
        """Record an observed latency measurement in milliseconds."""
        if latency_ms >= 0:
            self._samples.append(latency_ms)

    def get_metrics(self) -> LatencyMetrics:
        """Calculate p50, p95, p99 percentiles from recent sample window."""
        if not self._samples:
            return LatencyMetrics(
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                sample_count=0,
                min_ms=0.0,
                max_ms=0.0,
            )

        arr = np.array(self._samples)
        return LatencyMetrics(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            sample_count=len(arr),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )


class SystemHealthReport(BaseModel):
    """Structured health check report."""

    is_healthy: bool
    can_trade: bool
    status: str
    reasons: list[str] = Field(default_factory=list)
    connector_health: dict[str, Any] = Field(default_factory=dict)
    data_quality_healthy: bool = True
    breaker_tripped: bool = False
    breaker_reason: str | None = None
    latency: LatencyMetrics
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SystemHealthMonitor:
    """Enterprise health orchestrator aggregating connection, latency, and data quality states."""

    def __init__(self, latency_tracker: LatencyTracker | None = None) -> None:
        self.latency_tracker = latency_tracker or LatencyTracker()

    def generate_report(
        self,
        is_connected: bool,
        is_breaker_tripped: bool,
        breaker_reason: str | None = None,
        connector_health: dict[str, Any] | None = None,
    ) -> SystemHealthReport:
        """Evaluate overall health and whether trading decisions are safe."""
        reasons: list[str] = []
        is_healthy = True
        can_trade = True

        if not is_connected:
            is_healthy = False
            can_trade = False
            reasons.append("Market data connection is disconnected or interrupted.")

        if is_breaker_tripped:
            can_trade = False
            reasons.append(f"Data quality circuit breaker is tripped: {breaker_reason}")

        latency = self.latency_tracker.get_metrics()
        # High latency warning threshold (e.g. p99 > 500ms)
        if latency.p99_ms > 500.0:
            reasons.append(f"High processing latency detected: p99={latency.p99_ms:.1f}ms")

        status = "HEALTHY" if can_trade else ("DEGRADED" if is_healthy else "CRITICAL")

        return SystemHealthReport(
            is_healthy=is_healthy,
            can_trade=can_trade,
            status=status,
            reasons=reasons,
            connector_health=connector_health or {},
            data_quality_healthy=not is_breaker_tripped,
            breaker_tripped=is_breaker_tripped,
            breaker_reason=breaker_reason,
            latency=latency,
        )
