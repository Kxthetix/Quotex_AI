"""Incident Management Models (Section 47).

Defines lifecycle, state transitions, and audit records for production incidents:
OPEN -> INVESTIGATING -> MITIGATED -> RESOLVED -> CLOSED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class IncidentSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IncidentState(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    MITIGATED = "MITIGATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


@dataclass
class IncidentAction:
    timestamp: str
    actor: str
    action_taken: str
    outcome: str = "SUCCESS"
    notes: str = ""


@dataclass
class Incident:
    incident_id: str
    timestamp: str
    severity: IncidentSeverity
    component: str
    detected_by: str
    description: str
    state: IncidentState = IncidentState.OPEN
    actions: list[IncidentAction] = field(default_factory=list)
    resolution: str | None = None
    resolved_at: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["state"] = self.state.value
        return data
