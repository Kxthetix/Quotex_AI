"""Unit tests for Security, Secret Management, and RBAC Access Control (Sections 34, 35, 36, 38)."""

import os
from pathlib import Path

import pytest

from production.audit.audit_logger import AuditLogger
from production.security.access_control import (
    AccessControlManager,
    AccessDeniedError,
    Permission,
    Role,
    UserSession,
)
from production.security.secret_manager import SecretManager, SecretValidationError


def test_secret_manager_masking():
    assert SecretManager.mask_secret("very_secret_api_key_12345") == "ve****45"
    assert SecretManager.mask_secret("short") == "******"
    assert SecretManager.mask_secret(None) == "******"


def test_secret_manager_sanitize_connection_string():
    raw_uri = "postgresql://trader_user:super_secret_pw@db.prod.internal:5432/trading_db"
    sanitized = SecretManager.sanitize_connection_string(raw_uri)
    assert "super_secret_pw" not in sanitized
    assert sanitized == "postgresql://trader_user:***@db.prod.internal:5432/trading_db"


def test_secret_manager_production_validation():
    # Simulate production mode with missing secrets
    os.environ["TRADING_ENV"] = "production"
    # Ensure a non-existent secret is tested
    with pytest.raises(SecretValidationError):
        SecretManager.validate_required_secrets(required_keys=["NON_EXISTENT_SECRET_12345"])

    # Reset
    os.environ["TRADING_ENV"] = "development"


def test_access_control_read_only_monitor(tmp_path: Path):
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=audit_file)
    ac = AccessControlManager(audit_logger=logger)

    monitor_user = UserSession(user_id="monitor_bot", roles=[Role.READ_ONLY_MONITOR])

    # Allowed
    assert ac.authorize(monitor_user, Permission.VIEW_METRICS, resource="MetricsAPI") is True
    assert ac.authorize(monitor_user, Permission.VIEW_STATE, resource="PortfolioState") is True

    # Denied
    with pytest.raises(AccessDeniedError):
        ac.authorize(monitor_user, Permission.RESET_CIRCUIT_BREAKER, resource="RiskEngine")


def test_access_control_operator_and_model_manager(tmp_path: Path):
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=audit_file)
    ac = AccessControlManager(audit_logger=logger)

    operator = UserSession(user_id="op_dave", roles=[Role.OPERATOR])
    model_mgr = UserSession(user_id="mm_sarah", roles=[Role.MODEL_MANAGER])

    # Operator can reset breaker but cannot promote model
    assert ac.authorize(operator, Permission.RESET_CIRCUIT_BREAKER, resource="RiskEngine") is True
    with pytest.raises(AccessDeniedError):
        ac.authorize(operator, Permission.PROMOTE_MODEL, resource="ModelRegistry")

    # Model Manager can promote model but cannot reset breaker
    assert ac.authorize(model_mgr, Permission.PROMOTE_MODEL, resource="ModelRegistry") is True
    with pytest.raises(AccessDeniedError):
        ac.authorize(model_mgr, Permission.RESET_CIRCUIT_BREAKER, resource="RiskEngine")


def test_access_control_admin(tmp_path: Path):
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=audit_file)
    ac = AccessControlManager(audit_logger=logger)

    admin = UserSession(user_id="admin_root", roles=[Role.ADMIN])
    # Admin has all permissions
    assert ac.authorize(admin, Permission.CHANGE_RISK_LIMITS, resource="RiskConfig") is True
    assert ac.authorize(admin, Permission.MANAGE_SECRETS, resource="SecretVault") is True
