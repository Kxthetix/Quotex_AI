"""Production Metrics Registry and Collection (Sections 23, 24).

Implements thread-safe counters, gauges, and latency histograms for all
required system, application, trading, data, and model metrics.
"""

from __future__ import annotations

import collections
import contextlib
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator


@dataclass
class MetricValue:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Counter:
    """Monotonically increasing metric."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative amounts.")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Gauge:
    """Metric that can arbitrarily increase or decrease."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Histogram:
    """Latency and distribution measurement with summary stats."""

    def __init__(self, name: str, max_samples: int = 1000, description: str = "") -> None:
        self.name = name
        self.description = description
        self.max_samples = max_samples
        self._samples: collections.deque[float] = collections.deque(maxlen=max_samples)
        self._count = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)
            self._count += 1
            self._sum += value

    def summary(self) -> dict[str, float]:
        with self._lock:
            if not self._samples:
                return {"count": 0, "sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)
            p50 = sorted_samples[int(n * 0.50)]
            p95 = sorted_samples[min(int(n * 0.95), n - 1)]
            p99 = sorted_samples[min(int(n * 0.99), n - 1)]
            mean = self._sum / max(1, self._count)
            return {
                "count": float(self._count),
                "sum": self._sum,
                "mean": mean,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }


class MetricsRegistry:
    """Central metrics collector initialized with all mandatory Section 24 metrics."""

    _instance: MetricsRegistry | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}

        self._init_standard_metrics()

    @classmethod
    def get_instance(cls) -> MetricsRegistry:
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = MetricsRegistry()
            return cls._instance

    def _init_standard_metrics(self) -> None:
        """Register the exact mandatory metrics required by Section 24."""
        # Section 24 mandatory counters
        mandatory_counters = [
            "market_events_total",
            "market_events_invalid_total",
            "market_data_disconnects_total",
            "market_data_reconnects_total",
            "predictions_total",
            "signals_total",
            "orders_total",
            "fills_total",
            "trades_total",
            "errors_total",
            "circuit_breakers_total",
            "database_errors_total",
            "redis_errors_total",
        ]
        for name in mandatory_counters:
            self.counters[name] = Counter(name)

        # Section 24 mandatory latency histograms
        mandatory_histograms = [
            "processing_latency",
            "prediction_latency",
            "signal_latency",
            "risk_latency",
            "execution_latency",
        ]
        for name in mandatory_histograms:
            self.histograms[name] = Histogram(name)

        # Section 23 system & trading gauges
        mandatory_gauges = [
            "system_cpu_percent",
            "system_memory_mb",
            "system_disk_percent",
            "process_restart_count",
            "active_exposure",
            "current_drawdown",
            "data_freshness_seconds",
        ]
        for name in mandatory_gauges:
            self.gauges[name] = Gauge(name)

    def inc_counter(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name)
            self.counters[name].inc(amount)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name)
            self.gauges[name].set(value)

    def observe_histogram(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name)
            self.histograms[name].observe(value)

    @contextlib.contextmanager
    def time_block(self, histogram_name: str) -> Iterator[None]:
        """Context manager to measure latency of a code block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe_histogram(histogram_name, elapsed)

    def sample_system_metrics(self) -> None:
        """Sample OS resource utilization (disk, memory, cpu)."""
        # Disk usage of workspace
        try:
            usage = shutil.disk_usage(".")
            disk_pct = (usage.used / max(1, usage.total)) * 100.0
            self.set_gauge("system_disk_percent", disk_pct)
        except Exception:
            pass

        # Memory usage via psutil if available, or fallback
        try:
            import psutil
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            self.set_gauge("system_memory_mb", mem_mb)
            self.set_gauge("system_cpu_percent", psutil.cpu_percent(interval=None))
        except ImportError:
            pass

    def collect(self) -> dict[str, Any]:
        """Export all current metrics as a JSON-serializable snapshot."""
        self.sample_system_metrics()
        with self._lock:
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "counters": {k: c.value for k, c in self.counters.items()},
                "gauges": {k: g.value for k, g in self.gauges.items()},
                "histograms": {k: h.summary() for k, h in self.histograms.items()},
            }
