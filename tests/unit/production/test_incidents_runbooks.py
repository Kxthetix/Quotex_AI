"""Unit tests for Incident Management and Operational Runbooks (Sections 47, 48)."""

from pathlib import Path

from production.incidents.incident_manager import IncidentManager
from production.incidents.models import IncidentSeverity, IncidentState
from production.incidents.runbooks import RUNBOOKS, RunbookRegistry


def test_incident_manager_lifecycle(tmp_path: Path):
    log_file = tmp_path / "incidents.jsonl"
    im = IncidentManager(log_path=log_file)

    # 1. Create incident
    inc = im.create_incident(
        severity=IncidentSeverity.CRITICAL,
        component="Database",
        detected_by="HealthChecker",
        description="Connection pool timed out",
    )
    assert inc.state == IncidentState.OPEN
    assert len(im.get_open_incidents()) == 1

    # 2. Transition state
    im.transition_state(inc.incident_id, IncidentState.INVESTIGATING, actor="dave", notes="Checking logs")
    assert inc.state == IncidentState.INVESTIGATING

    # 3. Record action
    im.record_action(inc.incident_id, actor="dave", action_taken="RESTART_POSTGRES", notes="Restarted DB service")
    assert len(inc.actions) == 2

    # 4. Resolve incident
    im.resolve_incident(inc.incident_id, actor="dave", resolution="PostgreSQL container restarted, connections recovered.")
    assert inc.state == IncidentState.RESOLVED
    assert len(im.get_open_incidents()) == 0

    # 5. Verify persistence by creating a second instance reading the same file
    im2 = IncidentManager(log_path=log_file)
    restored_inc = im2.get_incident(inc.incident_id)
    assert restored_inc is not None
    assert restored_inc.state == IncidentState.RESOLVED
    assert restored_inc.resolution is not None


def test_runbooks_completeness():
    # Exactly 12 runbooks matching Section 48
    expected_runbooks = [
        "websocket_outage",
        "stale_market_data",
        "database_outage",
        "redis_outage",
        "model_failure",
        "model_artifact_corruption",
        "high_latency",
        "disk_full",
        "backup_failure",
        "circuit_breaker",
        "unexpected_model_behavior",
        "security_incident",
    ]

    for key in expected_runbooks:
        rb = RunbookRegistry.get_runbook(key)
        assert rb is not None, f"Missing required runbook: {key}"
        assert rb.id.startswith("RB-")
        assert len(rb.title) > 0
        assert len(rb.detection) > 0
        assert len(rb.immediate_safety_action) > 0
        assert len(rb.diagnosis) > 0
        assert len(rb.recovery) > 0
        assert len(rb.validation) > 0
        assert len(rb.return_to_service_criteria) > 0
