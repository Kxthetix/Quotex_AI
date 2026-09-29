"""Service Health and Readiness Models for Phase 9 Production Hardening.

Defines:
- Health states: HEALTHY, DEGRADED, BLOCKED, FAILED, UNKNOWN
- Component health check results
- System readiness evaluation structures
- Liveness and Dependency health structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ServiceHealthState(StrEnum):
    """Explicit health states for all system services (Section 4)."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class CheckType(StrEnum):
    """Check classification (Section 5)."""
    LIVENESS = "LIVENESS"
    READINESS = "READINESS"
    DEPENDENCY = "DEPENDENCY"


@dataclass
class ComponentHealthResult:
    """Outcome of a health check on an individual service or component."""
    component_name: str
    state: ServiceHealthState
    check_type: CheckType
    is_ready: bool
    details: str = ""
    latency_ms: float = 0.0
    mandatory: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def message(self) -> str:
        """Alias for details."""
        return self.details

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "state": str(self.state),
            "check_type": str(self.check_type),
            "is_ready": self.is_ready,
            "details": self.details,
            "latency_ms": round(self.latency_ms, 2),
            "mandatory": self.mandatory,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemReadinessReport:
    """Comprehensive readiness assessment determining if system can trade (Section 6)."""
    is_ready: bool
    is_alive: bool
    master_state: ServiceHealthState
    unhealthy_components: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    component_results: dict[str, ComponentHealthResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def blocking_issues(self) -> list[str]:
        """Alias for reasons list of blocking issues."""
        return self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "is_alive": self.is_alive,
            "master_state": str(self.master_state),
            "unhealthy_components": self.unhealthy_components,
            "reasons": self.reasons,
            "components": {k: v.to_dict() for k, v in self.component_results.items()},
            "timestamp": self.timestamp.isoformat(),
        }
