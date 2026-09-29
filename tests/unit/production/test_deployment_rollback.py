"""Unit tests for Deployment Preflight Validation and Rollback (Sections 37, 39, 41, 42)."""

from pathlib import Path

from production.audit.audit_logger import AuditLogger
from production.deployment.rollback import DeploymentRollbackManager
from production.deployment.validator import DeploymentValidator


def test_deployment_validator_passes():
    valid_cfg = {
        "features": {"window_sizes": [5, 15, 60]},
        "signal": {"threshold": 0.65},
        "risk": {"max_drawdown_limit": 0.15, "daily_loss_limit": 0.03},
    }

    res = DeploymentValidator.validate_preflight(
        config=valid_cfg,
        db_reachable=True,
        redis_reachable=True,
    )
    assert res.passed is True
    assert len(res.errors) == 0


def test_deployment_validator_catches_invalid_risk_and_missing_sections():
    bad_cfg = {
        "features": {},
        # missing "signal"
        "risk": {"max_drawdown_limit": 0.90, "daily_loss_limit": 0.05},  # invalid max_drawdown > 0.50
    }

    res = DeploymentValidator.validate_preflight(
        config=bad_cfg,
        db_reachable=False,
        redis_reachable=True,
    )
    assert res.passed is False
    assert any("signal" in err for err in res.errors)
    assert any("max_drawdown_limit" in err for err in res.errors)
    assert any("Database" in err for err in res.errors)


def test_deployment_rollback_cycle(tmp_path: Path):
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=audit_file)

    catalog = tmp_path / "releases.jsonl"
    rm = DeploymentRollbackManager(catalog_path=catalog, audit_logger=logger)

    cfg_v1 = {"version": "1.0", "risk": {"max_drawdown_limit": 0.10}}
    cfg_v2 = {"version": "2.0", "risk": {"max_drawdown_limit": 0.12}}

    # Deploy release 1
    rel1 = rm.register_deployment(
        app_version="1.0.0",
        model_version="model_v1",
        config=cfg_v1,
        deployed_by="ci_cd",
    )
    assert rm.active_release.release_id == rel1.release_id

    # Deploy release 2
    rel2 = rm.register_deployment(
        app_version="2.0.0",
        model_version="model_v2",
        config=cfg_v2,
        deployed_by="ci_cd",
    )
    assert rm.active_release.release_id == rel2.release_id

    # Roll back release 2
    success, restored_rel, msg = rm.rollback(actor="operator_1", reason="Degraded inference latency detected")
    assert success is True
    assert restored_rel is not None
    assert restored_rel.app_version == "1.0.0"
    assert restored_rel.model_version == "model_v1"
    assert rm.active_release.model_version == "model_v1"

    # Verify audit trail logged both deployment and rollback
    valid, audit_msg = logger.verify_integrity()
    assert valid is True
