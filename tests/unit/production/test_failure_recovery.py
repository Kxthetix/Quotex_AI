"""Unit tests for Failure Classification, Retry, State Recovery, and Shutdown (Sections 7, 8, 9, 14, 15, 16, 43)."""

import pytest

from production.recovery.failure_classifier import (
    FailureCategory,
    FailureClassifier,
    RecoveryAction,
)
from production.recovery.retry import RetryPolicy
from production.recovery.shutdown import GracefulShutdownManager
from production.recovery.state_recovery import (
    PersistenceBuffer,
    RestoredSystemState,
    StateRecoveryManager,
)


def test_failure_classifier_categories():
    classifier = FailureClassifier()

    # Transient network timeout
    assessment = classifier.classify(TimeoutError("Connection timed out"))
    assert assessment.severity == FailureCategory.TRANSIENT
    assert assessment.recommended_action == RecoveryAction.RETRY

    # Connection reset (recoverable)
    assessment = classifier.classify(ConnectionResetError("Socket reset by peer"))
    assert assessment.severity == FailureCategory.RECOVERABLE
    assert assessment.recommended_action == RecoveryAction.RECONNECT

    # Corrupted data (critical)
    assessment = classifier.classify(ValueError("Corrupted candle data received"))
    assert assessment.severity in (FailureCategory.CRITICAL, FailureCategory.FATAL)
    assert assessment.recommended_action in (RecoveryAction.HALT_DECISIONS, RecoveryAction.SAFE_SHUTDOWN)


def test_retry_policy_exponential_backoff():
    policy = RetryPolicy(
        max_attempts=4,
        initial_delay=0.1,
        max_delay=1.0,
        backoff_factor=2.0,
        jitter=False,
    )

    attempts = 0

    def flaky_task():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Temporary glitch")
        return "SUCCESS"

    result = policy.execute(flaky_task)
    assert result == "SUCCESS"
    assert attempts == 3


def test_retry_policy_max_attempts_exceeded():
    policy = RetryPolicy(max_attempts=2, initial_delay=0.01, jitter=False)

    def persistently_failing():
        raise ConnectionRefusedError("Host unreachable")

    with pytest.raises(ConnectionRefusedError):
        policy.execute(persistently_failing)


def test_persistence_buffer_bounded_memory():
    # Max size 5
    buffer = PersistenceBuffer(max_buffer_size=5)

    for i in range(10):
        buffer.push({"record_id": i, "data": f"tick_{i}"})

    assert buffer.size() == 5
    assert buffer.dropped_records_count == 5

    # Test flush
    flushed_items = []

    def mock_db_writer(batch):
        flushed_items.extend(batch)
        return True

    count = buffer.flush(mock_db_writer)
    assert count == 5
    assert buffer.size() == 0
    assert len(flushed_items) == 5
    assert flushed_items[0]["record_id"] == 5


def test_state_recovery_manager():
    mgr = StateRecoveryManager()

    def mock_loader():
        return {
            "session_id": "sess_12345",
            "active_model_version": "v2.1.0",
            "portfolio_equity": 150_000.0,
            "portfolio_cash": 120_000.0,
            "open_positions_count": 2,
        }

    restored = mgr.restore_state(mock_loader)
    assert restored.session_id == "sess_12345"
    assert restored.active_model_version == "v2.1.0"
    assert restored.portfolio_equity == 150_000.0
    assert restored.open_positions_count == 2


def test_graceful_shutdown_manager():
    shutdown_mgr = GracefulShutdownManager()
    events = []

    shutdown_mgr.register_hook("stop_trading", lambda: events.append("orders_stopped"))
    shutdown_mgr.register_hook("flush_buffers", lambda: events.append("buffers_flushed"))
    shutdown_mgr.register_hook("close_db", lambda: events.append("db_closed"))

    report = shutdown_mgr.shutdown(reason="Scheduled Maintenance")
    assert report.success is True
    assert events == ["orders_stopped", "buffers_flushed", "db_closed"]
