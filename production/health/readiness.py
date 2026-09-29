"""System Readiness Orchestrator (Section 6, 39).

Ensures the system only enters READY = True when all mandatory internal and external
dependencies pass health and readiness criteria.
"""

from __future__ import annotations

from typing import Any

from monitoring.logging import get_logger
from production.health.checker import HealthChecker
from production.health.models import (
    ComponentHealthResult,
    ServiceHealthState,
    SystemReadinessReport,
)

logger = get_logger(__name__)


class ReadinessCoordinator:
    """Evaluates full system readiness to protect against premature live execution."""

    def __init__(
        self,
        checker: HealthChecker | None = None,
        health_checker: HealthChecker | None = None,
        mandatory_components: list[str] | None = None,
    ) -> None:
        self.checker = checker or health_checker
        if self.checker is None:
            raise ValueError("HealthChecker instance must be provided.")
        self.mandatory_components = mandatory_components

    def evaluate_readiness(self) -> SystemReadinessReport:
        """Run all probes and compile comprehensive readiness report."""
        results = self.checker.check_all()

        unhealthy_mandatory: list[str] = []
        reasons: list[str] = []
        is_alive = True

        for name, res in results.items():
            # Check liveness
            if res.check_type == "LIVENESS" and res.state in (ServiceHealthState.FAILED, ServiceHealthState.BLOCKED):
                is_alive = False

            # Check mandatory readiness
            is_mand = (name in self.mandatory_components) if self.mandatory_components is not None else res.mandatory
            if is_mand and res.state not in (ServiceHealthState.HEALTHY,):
                unhealthy_mandatory.append(name)
                reasons.append(f"Mandatory component '{name}' is in state {res.state}: {res.details}")

        is_ready = len(unhealthy_mandatory) == 0

        # Determine master health state
        if not is_alive:
            master_state = ServiceHealthState.FAILED
        elif any(r.state == ServiceHealthState.BLOCKED for r in results.values()):
            master_state = ServiceHealthState.BLOCKED
        elif any(r.state == ServiceHealthState.FAILED for r in results.values()):
            master_state = ServiceHealthState.FAILED
        elif any(r.state == ServiceHealthState.DEGRADED for r in results.values()) or not is_ready:
            master_state = ServiceHealthState.DEGRADED
        else:
            master_state = ServiceHealthState.HEALTHY

        report = SystemReadinessReport(
            is_ready=is_ready,
            is_alive=is_alive,
            master_state=master_state,
            unhealthy_components=unhealthy_mandatory,
            reasons=reasons,
            component_results=results,
        )

        if not is_ready:
            logger.warning("system_not_ready", master_state=str(master_state), reasons=reasons)
        else:
            logger.info("system_ready_for_execution", master_state=str(master_state))

        return report
