"""Production Alerting System (Section 25).

Classifies, deduplicates, and dispatches CRITICAL, WARNING, and INFO alerts.
Enforces cooldown periods to prevent alert storms during cascading failures.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    component: str
    title: str
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


class AlertManager:
    """Manages alert routing, cooldown throttling, and subscriber notifications."""

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: dict[str, float] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._history: list[Alert] = []
        self._subscribers: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Register an alert sink / handler."""
        with self._lock:
            self._subscribers.append(callback)

    def trigger(
        self,
        severity: AlertSeverity,
        component: str,
        title: str,
        description: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        bypass_cooldown: bool = False,
    ) -> Alert | None:
        """Trigger an alert. Returns the Alert if dispatched, or None if throttled."""
        dedup_key = f"{severity.value}:{component}:{title}"
        now = time.time()

        with self._lock:
            if not bypass_cooldown:
                last_time = self._last_fired.get(dedup_key, 0.0)
                if now - last_time < self.cooldown_seconds:
                    # Suppressed by cooldown
                    return None

            self._last_fired[dedup_key] = now
            alert_id = f"alt_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self._history)}"
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                component=component,
                title=title,
                description=description,
                correlation_id=correlation_id,
                metadata=metadata or {},
            )

            if severity == AlertSeverity.CRITICAL:
                self._active_alerts[alert_id] = alert

            self._history.append(alert)
            # Retain maximum 1000 historical alerts in memory
            if len(self._history) > 1000:
                self._history.pop(0)

            subscribers_to_notify = list(self._subscribers)

        # Notify subscribers outside lock
        for sub in subscribers_to_notify:
            try:
                sub(alert)
            except Exception:
                pass

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an active critical alert as resolved."""
        with self._lock:
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]
                return True
            return False

    def get_active_critical_alerts(self) -> list[Alert]:
        """Return all currently unresolved critical alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_history(self) -> list[Alert]:
        with self._lock:
            return list(self._history)
