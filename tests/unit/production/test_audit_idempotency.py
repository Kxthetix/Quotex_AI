"""Unit tests for Audit Trail, Structured Logging, Correlation IDs, and Idempotency (Sections 17, 19, 20, 21, 22)."""

import json
from pathlib import Path

import pytest

from production.audit.audit_logger import (
    AuditLogger,
    StructuredLogger,
    redact_dict,
    redact_sensitive_str,
)
from production.audit.correlation import CorrelationChain, CorrelationTracker
from production.audit.idempotency import IdempotencyError, IdempotencyManager


def test_correlation_tracker_chain_propagation():
    chain = CorrelationTracker.create_chain(market_event_id="mkt_1001")
    assert chain.market_event_id == "mkt_1001"
    assert chain.correlation_id.startswith("corr_")

    CorrelationTracker.link_prediction("pred_2001")
    CorrelationTracker.link_signal("sig_3001")
    CorrelationTracker.link_risk_decision("risk_4001")
    CorrelationTracker.link_order("ord_5001")
    CorrelationTracker.link_fill("fill_6001")
    CorrelationTracker.link_trade("trd_7001")

    current = CorrelationTracker.get_current_chain()
    assert current.prediction_id == "pred_2001"
    assert current.signal_id == "sig_3001"
    assert current.risk_decision_id == "risk_4001"
    assert current.order_id == "ord_5001"
    assert current.fill_id == "fill_6001"
    assert current.trade_id == "trd_7001"

    data = current.to_dict()
    assert data["trade_id"] == "trd_7001"


def test_redact_sensitive_data():
    raw_text = "Connecting with api_key='secret12345678' and bearer token eyJhbGciOiJIUzI1NiIsInR5cCI"
    cleaned = redact_sensitive_str(raw_text)
    assert "secret12345678" not in cleaned
    assert "***REDACTED***" in cleaned

    payload = {
        "user": "alice",
        "api_key": "my_secret_key_123",
        "nested": {"password": "admin_password", "amount": 100},
    }
    safe_dict = redact_dict(payload)
    assert safe_dict["api_key"] == "***REDACTED***"
    assert safe_dict["nested"]["password"] == "***REDACTED***"
    assert safe_dict["nested"]["amount"] == 100


def test_audit_logger_hash_chain_and_verification(tmp_path: Path):
    audit_file = tmp_path / "audit_trail.jsonl"
    logger = AuditLogger(log_path=audit_file)

    rec1 = logger.record_action(
        actor="operator_1",
        action="CIRCUIT_BREAKER_RESET",
        resource="RiskEngine",
        reason="Market volatility normalized",
    )
    assert rec1.prev_hash == "GENESIS"
    assert rec1.record_hash != ""

    rec2 = logger.record_action(
        actor="model_manager",
        action="MODEL_PROMOTION",
        resource="ModelRegistry",
        reason="Candidate passed all 5 validation gates",
    )
    assert rec2.prev_hash == rec1.record_hash

    valid, msg = logger.verify_integrity()
    assert valid is True
    assert "2 records valid" in msg


def test_audit_logger_detects_tampering(tmp_path: Path):
    audit_file = tmp_path / "audit_trail.jsonl"
    logger = AuditLogger(log_path=audit_file)

    logger.record_action(actor="alice", action="ACT_1", resource="res1", reason="r1")
    logger.record_action(actor="bob", action="ACT_2", resource="res2", reason="r2")

    # Tamper with file
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    data = json.loads(lines[0])
    data["actor"] = "malicious_actor"
    lines[0] = json.dumps(data)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Re-verify
    valid, msg = logger.verify_integrity()
    assert valid is False
    assert "Tampered record" in msg or "Broken hash chain" in msg


def test_idempotency_manager_deduplication():
    mgr = IdempotencyManager(max_records=100)
    key = mgr.generate_key("order", "BTC/USDT", 1.5, "BUY")

    executions = 0

    def place_order():
        nonlocal executions
        executions += 1
        return {"order_id": "ord_999", "status": "FILLED"}

    res1, executed1 = mgr.execute_idempotent(key, "ORDER_PLACEMENT", place_order)
    assert executed1 is True
    assert executions == 1
    assert res1["order_id"] == "ord_999"

    # Second call with same key must NOT re-execute
    res2, executed2 = mgr.execute_idempotent(key, "ORDER_PLACEMENT", place_order)
    assert executed2 is False
    assert executions == 1
    assert res2["order_id"] == "ord_999"


def test_idempotency_in_progress_protection():
    mgr = IdempotencyManager()
    mgr.mark_in_progress("key_abc", "MUTATION")

    with pytest.raises(IdempotencyError):
        mgr.mark_in_progress("key_abc", "MUTATION")
