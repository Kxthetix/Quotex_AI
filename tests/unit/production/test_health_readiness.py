"""Unit tests for Health Checks and System Readiness (Sections 4, 5, 6)."""

import pytest

from production.health.checker import HealthChecker
from production.health.models import (
    CheckType,
    ComponentHealthResult,
    ServiceHealthState,
)
from production.health.readiness import ReadinessCoordinator


def test_health_checker_registration_and_probing():
    checker = HealthChecker()
    checker.register_check(
        component="MarketData",
        check_type=CheckType.LIVENESS,
        probe_fn=lambda: (ServiceHealthState.HEALTHY, "WebSocket responsive"),
    )
    checker.register_check(
        component="Database",
        check_type=CheckType.DEPENDENCY,
        probe_fn=lambda: (ServiceHealthState.HEALTHY, "Postgres ping OK"),
    )

    results = checker.run_all_checks()
    assert len(results) == 2

    md_res = checker.run_component_check("MarketData", CheckType.LIVENESS)
    assert md_res is not None
    assert md_res.state == ServiceHealthState.HEALTHY
    assert "responsive" in md_res.message
    assert md_res.latency_ms >= 0.0


def test_health_checker_failure_handling():
    checker = HealthChecker()

    def failing_probe():
        raise ConnectionResetError("Remote server closed connection")

    checker.register_check(
        component="Redis",
        check_type=CheckType.DEPENDENCY,
        probe_fn=failing_probe,
    )

    res = checker.run_component_check("Redis", CheckType.DEPENDENCY)
    assert res is not None
    assert res.state == ServiceHealthState.FAILED
    assert "Remote server closed connection" in res.message


def test_readiness_coordinator_all_healthy():
    checker = HealthChecker()
    checker.register_check("MarketData", CheckType.READINESS, lambda: (ServiceHealthState.HEALTHY, "OK"))
    checker.register_check("Database", CheckType.DEPENDENCY, lambda: (ServiceHealthState.HEALTHY, "OK"))
    checker.register_check("Redis", CheckType.DEPENDENCY, lambda: (ServiceHealthState.HEALTHY, "OK"))
    checker.register_check("ModelService", CheckType.READINESS, lambda: (ServiceHealthState.HEALTHY, "OK"))
    checker.register_check("RiskEngine", CheckType.READINESS, lambda: (ServiceHealthState.HEALTHY, "OK"))

    coordinator = ReadinessCoordinator(
        health_checker=checker,
        mandatory_components=["MarketData", "Database", "Redis", "ModelService", "RiskEngine"],
    )

    report = coordinator.evaluate_readiness()
    assert report.is_ready is True
    assert len(report.blocking_issues) == 0


def test_readiness_coordinator_blocks_when_mandatory_component_fails():
    checker = HealthChecker()
    checker.register_check("MarketData", CheckType.READINESS, lambda: (ServiceHealthState.HEALTHY, "OK"))
    checker.register_check("Database", CheckType.DEPENDENCY, lambda: (ServiceHealthState.FAILED, "DB connection pool exhausted"))
    checker.register_check("Redis", CheckType.DEPENDENCY, lambda: (ServiceHealthState.HEALTHY, "OK"))

    coordinator = ReadinessCoordinator(
        health_checker=checker,
        mandatory_components=["MarketData", "Database", "Redis"],
    )

    report = coordinator.evaluate_readiness()
    assert report.is_ready is False
    assert len(report.blocking_issues) == 1
    assert "Database" in report.blocking_issues[0]
