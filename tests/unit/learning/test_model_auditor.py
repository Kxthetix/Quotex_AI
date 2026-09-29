"""Tests for Model Auditor (Phase 8 Sections 8, 9, 10, 16, 17, 39)."""

from datetime import UTC, datetime, timedelta
import pytest

from learning.model_governance.auditor import AuditorConfig, ModelAuditor
from learning.model_governance.models import AuditRecommendation
from paper_trading.models import PredictionRecord, PredictionStatus


def test_auditor_insufficient_sample_size() -> None:
    auditor = ModelAuditor(config=AuditorConfig(min_audit_sample_size=30))
    now = datetime.now(UTC)

    # Only 10 predictions (less than min 30)
    preds = [
        PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=10 - i),
            model_version="dir_v001",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT",
            confidence=0.7,
        )
        for i in range(10)
    ]

    result = auditor.audit_model(model_version="dir_v001", predictions=preds)
    # Must NOT classify as failed or degraded simply because of small sample
    assert result.recommendation_state == AuditRecommendation.INSUFFICIENT_DATA
    assert "below minimum requirement" in result.warnings[0]


def test_auditor_healthy_model() -> None:
    auditor = ModelAuditor(config=AuditorConfig(min_audit_sample_size=30, accuracy_warning_threshold=0.52))
    now = datetime.now(UTC)

    # 40 predictions with 75% accuracy
    preds = []
    for i in range(40):
        is_corr = i < 30
        preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=40 - i),
            model_version="dir_v001",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if is_corr else "INCORRECT",
            confidence=0.85 if is_corr else 0.20,
            prediction_error=0.005 if is_corr else 0.02,
            regime="TREND_UP",
        ))

    result = auditor.audit_model(model_version="dir_v001", predictions=preds)
    assert result.recommendation_state == AuditRecommendation.HEALTHY
    assert result.performance_metrics["accuracy"] == 0.75
    assert "TREND_UP" in result.regime_metrics
    assert result.regime_metrics["TREND_UP"]["accuracy"] == 0.75


def test_auditor_degraded_performance_triggers_retrain() -> None:
    auditor = ModelAuditor(config=AuditorConfig(
        min_audit_sample_size=30,
        accuracy_critical_threshold=0.48,
    ))
    now = datetime.now(UTC)

    # 50 predictions with 30% accuracy (severe drop)
    preds = []
    for i in range(50):
        is_corr = i < 15
        preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=50 - i),
            model_version="dir_v001",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if is_corr else "INCORRECT",
            confidence=0.60,
            prediction_error=0.03,
            regime="SIDEWAYS",
        ))

    result = auditor.audit_model(model_version="dir_v001", predictions=preds)
    assert result.recommendation_state == AuditRecommendation.RETRAIN_CANDIDATE
    assert result.performance_metrics["accuracy"] == 0.30
    assert any("below critical threshold" in w for w in result.warnings)
