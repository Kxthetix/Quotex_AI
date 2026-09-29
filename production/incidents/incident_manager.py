"""Incident Manager (Section 47).

Manages lifecycle transitions, containment logging, and durable audit records
for operational disruptions and safety interventions.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from production.audit.correlation import CorrelationTracker
from production.incidents.models import (
    Incident,
    IncidentAction,
    IncidentSeverity,
    IncidentState,
)


class IncidentManager:
    """Tracks and records production incidents with durable JSONL audit log."""

    def __init__(self, log_path: Path | str = "data/incidents/incidents.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._incidents: dict[str, Incident] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load incidents from disk if log exists."""
        if not self.log_path.exists():
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    actions = [IncidentAction(**a) for a in data.get("actions", [])]
                    inc = Incident(
                        incident_id=data["incident_id"],
                        timestamp=data["timestamp"],
                        severity=IncidentSeverity(data["severity"]),
                        component=data["component"],
                        detected_by=data["detected_by"],
                        description=data["description"],
                        state=IncidentState(data["state"]),
                        actions=actions,
                        resolution=data.get("resolution"),
                        resolved_at=data.get("resolved_at"),
                        correlation_id=data.get("correlation_id"),
                    )
                    self._incidents[inc.incident_id] = inc
                except Exception:
                    continue

    def _persist(self, incident: Incident) -> None:
        """Append incident state snapshot to log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident.to_dict()) + "\n")

    def create_incident(
        self,
        severity: IncidentSeverity,
        component: str,
        detected_by: str,
        description: str,
        correlation_id: str | None = None,
    ) -> Incident:
        """Create and record a new production incident."""
        with self._lock:
            ts = datetime.now(UTC)
            incident_id = f"inc_{ts.strftime('%Y%m%d_%H%M%S')}_{len(self._incidents) + 1}"
            if correlation_id is None:
                correlation_id = CorrelationTracker.get_current_chain().correlation_id

            incident = Incident(
                incident_id=incident_id,
                timestamp=ts.isoformat(),
                severity=severity,
                component=component,
                detected_by=detected_by,
                description=description,
                state=IncidentState.OPEN,
                correlation_id=correlation_id,
            )
            self._incidents[incident_id] = incident
            self._persist(incident)
            return incident

    def transition_state(
        self,
        incident_id: str,
        new_state: IncidentState,
        actor: str,
        notes: str = "",
    ) -> Incident:
        """Transition incident state and log action."""
        with self._lock:
            if incident_id not in self._incidents:
                raise KeyError(f"Incident '{incident_id}' not found.")
            incident = self._incidents[incident_id]
            incident.state = new_state
            incident.actions.append(
                IncidentAction(
                    timestamp=datetime.now(UTC).isoformat(),
                    actor=actor,
                    action_taken=f"STATE_TRANSITION:{new_state.value}",
                    notes=notes,
                )
            )
            self._persist(incident)
            return incident

    def record_action(
        self,
        incident_id: str,
        actor: str,
        action_taken: str,
        outcome: str = "SUCCESS",
        notes: str = "",
    ) -> Incident:
        """Log a diagnostic or recovery action taken during incident response."""
        with self._lock:
            if incident_id not in self._incidents:
                raise KeyError(f"Incident '{incident_id}' not found.")
            incident = self._incidents[incident_id]
            incident.actions.append(
                IncidentAction(
                    timestamp=datetime.now(UTC).isoformat(),
                    actor=actor,
                    action_taken=action_taken,
                    outcome=outcome,
                    notes=notes,
                )
            )
            self._persist(incident)
            return incident

    def resolve_incident(
        self,
        incident_id: str,
        actor: str,
        resolution: str,
    ) -> Incident:
        """Mark incident as resolved with resolution notes."""
        with self._lock:
            if incident_id not in self._incidents:
                raise KeyError(f"Incident '{incident_id}' not found.")
            incident = self._incidents[incident_id]
            incident.state = IncidentState.RESOLVED
            incident.resolution = resolution
            incident.resolved_at = datetime.now(UTC).isoformat()
            incident.actions.append(
                IncidentAction(
                    timestamp=incident.resolved_at,
                    actor=actor,
                    action_taken="RESOLVED",
                    notes=resolution,
                )
            )
            self._persist(incident)
            return incident

    def close_incident(self, incident_id: str, actor: str, notes: str = "") -> Incident:
        """Close an incident after post-incident review."""
        return self.transition_state(incident_id, IncidentState.CLOSED, actor=actor, notes=notes)

    def get_open_incidents(self) -> list[Incident]:
        """Retrieve all currently active / non-closed incidents."""
        with self._lock:
            return [
                inc for inc in self._incidents.values()
                if inc.state not in (IncidentState.RESOLVED, IncidentState.CLOSED)
            ]

    def get_incident(self, incident_id: str) -> Incident | None:
        with self._lock:
            return self._incidents.get(incident_id)
