"""Deployment Rollback and Release Management (Section 42).

Maintains version history of deployments (application, configuration, model version)
and provides automated atomic rollback to known-good states when a deployment fails.
"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


@dataclass
class ReleaseSnapshot:
    release_id: str
    app_version: str
    model_version: str
    config_snapshot: dict[str, Any]
    deployed_at: str
    deployed_by: str
    status: str = "ACTIVE"  # ACTIVE, SUPERSEDED, ROLLED_BACK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeploymentRollbackManager:
    """Tracks deployed releases and manages atomic rollback to previous releases."""

    def __init__(
        self,
        catalog_path: Path | str = "data/deployment/releases.jsonl",
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger
        self._history: list[ReleaseSnapshot] = []
        self._active_release: ReleaseSnapshot | None = None
        self._lock = threading.Lock()
        self._load_catalog()

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rel = ReleaseSnapshot(**data)
                    self._history.append(rel)
                    if rel.status == "ACTIVE":
                        self._active_release = rel
                except Exception:
                    continue

    def _persist(self, release: ReleaseSnapshot) -> None:
        with open(self.catalog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(release.to_dict()) + "\n")

    def register_deployment(
        self,
        app_version: str,
        model_version: str,
        config: dict[str, Any],
        deployed_by: str,
    ) -> ReleaseSnapshot:
        """Register a new production deployment."""
        with self._lock:
            ts = datetime.now(UTC)
            release_id = f"rel_{ts.strftime('%Y%m%d_%H%M%S')}_{len(self._history) + 1}"

            # Mark previously active release as superseded
            if self._active_release:
                self._active_release.status = "SUPERSEDED"

            new_release = ReleaseSnapshot(
                release_id=release_id,
                app_version=app_version,
                model_version=model_version,
                config_snapshot=copy.deepcopy(config),
                deployed_at=ts.isoformat(),
                deployed_by=deployed_by,
                status="ACTIVE",
            )
            self._history.append(new_release)
            self._active_release = new_release
            self._persist(new_release)

            if self.audit_logger:
                self.audit_logger.record_action(
                    actor=deployed_by,
                    action="DEPLOYMENT_PROMOTED",
                    resource=f"release:{release_id}",
                    reason=f"Deployed app {app_version} with model {model_version}",
                    previous_state=None,
                    new_state=new_release.to_dict(),
                )

            return new_release

    def rollback(self, actor: str, reason: str) -> tuple[bool, ReleaseSnapshot | None, str]:
        """Roll back to the previous known-good release."""
        with self._lock:
            if len(self._history) < 2:
                return False, None, "No previous release found in deployment history to rollback to."

            # Find latest superseded release
            candidates = [
                r for r in self._history
                if r.status == "SUPERSEDED" and r != self._active_release
            ]
            if not candidates:
                return False, None, "No valid superseded release available for rollback."

            previous_release = candidates[-1]
            old_release = self._active_release

            if old_release:
                old_release.status = "ROLLED_BACK"
                self._persist(old_release)

            # Re-activate previous release as a new active release entry
            ts = datetime.now(UTC)
            rollback_release = ReleaseSnapshot(
                release_id=f"rel_rollback_{ts.strftime('%Y%m%d_%H%M%S')}",
                app_version=previous_release.app_version,
                model_version=previous_release.model_version,
                config_snapshot=copy.deepcopy(previous_release.config_snapshot),
                deployed_at=ts.isoformat(),
                deployed_by=f"ROLLBACK:{actor}",
                status="ACTIVE",
            )
            self._history.append(rollback_release)
            self._active_release = rollback_release
            self._persist(rollback_release)

            if self.audit_logger:
                self.audit_logger.record_action(
                    actor=actor,
                    action="DEPLOYMENT_ROLLED_BACK",
                    resource=f"release:{rollback_release.release_id}",
                    reason=reason,
                    previous_state=old_release.to_dict() if old_release else None,
                    new_state=rollback_release.to_dict(),
                )

            msg = (
                f"Successfully rolled back from {old_release.release_id if old_release else 'None'} "
                f"to {previous_release.release_id} (App: {previous_release.app_version}, "
                f"Model: {previous_release.model_version})"
            )
            return True, rollback_release, msg

    @property
    def active_release(self) -> ReleaseSnapshot | None:
        with self._lock:
            return self._active_release

    def get_history(self) -> list[ReleaseSnapshot]:
        with self._lock:
            return list(self._history)
