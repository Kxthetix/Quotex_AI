"""Health Checker implementing Liveness, Readiness, and Dependency Probes (Sections 4, 5, 39)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Callable

from monitoring.logging import get_logger
from production.health.models import (
    CheckType,
    ComponentHealthResult,
    ServiceHealthState,
)

logger = get_logger(__name__)


class HealthChecker:
    """Probes components across Liveness, Readiness, and External Dependencies."""

    def __init__(self) -> None:
        self._probes: dict[str, tuple[CheckType, bool, Callable[[], tuple[ServiceHealthState, str, dict[str, Any]]]]] = {}

    def register_probe(
        self,
        component_name: str,
        check_type: CheckType,
        probe_func: Callable[[], Any],
        mandatory: bool = True,
    ) -> None:
        """Register a health probe for a component."""
        self._probes[component_name] = (check_type, mandatory, probe_func)

    def register_check(
        self,
        component: str,
        check_type: CheckType,
        probe_fn: Callable[[], Any],
        mandatory: bool = True,
    ) -> None:
        """Alias for register_probe."""
        self.register_probe(component, check_type, probe_fn, mandatory)

    def check_component(self, component_name: str, check_type_filter: CheckType | None = None) -> ComponentHealthResult:
        """Execute a probe for an individual component."""
        if component_name not in self._probes:
            return ComponentHealthResult(
                component_name=component_name,
                state=ServiceHealthState.UNKNOWN,
                check_type=check_type_filter or CheckType.READINESS,
                is_ready=False,
                details=f"No probe registered for {component_name}",
            )

        check_type, mandatory, probe_func = self._probes[component_name]
        start_time = time.perf_counter()
        try:
            raw_res = probe_func()
            metadata = {}
            if isinstance(raw_res, tuple):
                if len(raw_res) == 3:
                    state, details, metadata = raw_res
                elif len(raw_res) == 2:
                    state, details = raw_res
                else:
                    state = raw_res[0]
                    details = "Probe executed"
            else:
                state = raw_res
                details = "Probe executed"

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            is_ready = state in (ServiceHealthState.HEALTHY, ServiceHealthState.DEGRADED) and not mandatory or state == ServiceHealthState.HEALTHY
            return ComponentHealthResult(
                component_name=component_name,
                state=state,
                check_type=check_type,
                is_ready=is_ready,
                details=details,
                latency_ms=elapsed_ms,
                mandatory=mandatory,
                metadata=metadata,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("health_probe_failed", component=component_name, error=str(e))
            return ComponentHealthResult(
                component_name=component_name,
                state=ServiceHealthState.FAILED,
                check_type=check_type,
                is_ready=False,
                details=f"Probe exception: {e}",
                latency_ms=elapsed_ms,
                mandatory=mandatory,
            )

    def run_component_check(self, component_name: str, check_type: CheckType | None = None) -> ComponentHealthResult:
        """Alias for check_component."""
        return self.check_component(component_name, check_type)

    def run_all_checks(self) -> dict[str, ComponentHealthResult]:
        """Alias for check_all."""
        return self.check_all()

    def check_all(self) -> dict[str, ComponentHealthResult]:
        """Execute all registered probes."""
        return {name: self.check_component(name) for name in self._probes}

    def check_liveness(self) -> dict[str, ComponentHealthResult]:
        """Execute only liveness probes."""
        return {
            name: self.check_component(name)
            for name, (ctype, _, _) in self._probes.items()
            if ctype == CheckType.LIVENESS
        }

    def check_dependencies(self) -> dict[str, ComponentHealthResult]:
        """Execute external dependency probes."""
        return {
            name: self.check_component(name)
            for name, (ctype, _, _) in self._probes.items()
            if ctype == CheckType.DEPENDENCY
        }
