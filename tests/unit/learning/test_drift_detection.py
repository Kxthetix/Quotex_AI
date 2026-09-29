"""Tests for Drift Detection (Phase 8 Sections 11, 12, 13, 14, 15, 40)."""

from datetime import UTC, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from learning.model_governance.drift import (
    ConceptDriftDetector,
    FeatureDriftDetector,
    PredictionDriftDetector,
    RegimeDriftDetector,
)
from learning.model_governance.drift.feature_drift import calculate_psi
from learning.model_governance.models import DriftStatus
from paper_trading.models import PredictionRecord, PredictionStatus


def test_psi_calculation_nominal() -> None:
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=500)
    cur = np.random.normal(loc=0.05, scale=1.0, size=500)

    psi = calculate_psi(ref, cur)
    # Very minor shift should have PSI < 0.10
    assert psi < 0.10


def test_psi_calculation_severe_shift() -> None:
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=500)
    cur = np.random.normal(loc=2.5, scale=1.0, size=500)

    psi = calculate_psi(ref, cur)
    # Severe shift should have PSI > 0.25
    assert psi > 0.25


def test_feature_drift_detector() -> None:
    detector = FeatureDriftDetector(psi_moderate_threshold=0.10, psi_severe_threshold=0.25)
    np.random.seed(42)

    ref_data = pd.Series(np.random.normal(0, 1, 200))
    # Moderate shift
    mod_data = pd.Series(np.random.normal(0.6, 1, 200))
    # Severe shift
    sev_data = pd.Series(np.random.normal(3.0, 1, 200))

    res_mod = detector.evaluate_feature("feat_mod", ref_data, mod_data)
    assert res_mod.status in (DriftStatus.MODERATE_DRIFT, DriftStatus.SEVERE_DRIFT)

    res_sev = detector.evaluate_feature("feat_sev", ref_data, sev_data)
    assert res_sev.status == DriftStatus.SEVERE_DRIFT

    # Insufficient samples
    small_data = pd.Series([1.0, 2.0])
    res_small = detector.evaluate_feature("feat_small", ref_data, small_data)
    assert res_small.status == DriftStatus.INSUFFICIENT_DATA


def test_prediction_drift_detector() -> None:
    detector = PredictionDriftDetector(min_samples=20)
    now = datetime.now(UTC)

    # Reference: mostly balanced BUY/SELL with moderate confidence
    ref_preds = []
    for i in range(50):
        ref_preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=100 - i),
            model_version="v1.0.0",
            signal_action="BUY" if i % 2 == 0 else "SELL",
            confidence=0.60 + (i % 5) * 0.02,
            expected_return=0.005,
        ))

    # Current: shifted to extreme BUY with high confidence
    cur_preds = []
    for i in range(50):
        cur_preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=50 - i),
            model_version="v1.0.0",
            signal_action="BUY",
            confidence=0.95,
            expected_return=0.03,
        ))

    report = detector.evaluate_predictions(ref_preds, cur_preds)
    assert report["status"] in (str(DriftStatus.MODERATE_DRIFT), str(DriftStatus.SEVERE_DRIFT))
    assert report["max_action_frequency_shift"] >= 0.20


def test_concept_drift_detector() -> None:
    detector = ConceptDriftDetector(min_evaluated_samples=20)
    now = datetime.now(UTC)

    # Reference: high accuracy (80%)
    ref_preds = []
    for i in range(40):
        is_correct = i < 32
        ref_preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=100 - i),
            model_version="v1.0.0",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if is_correct else "INCORRECT",
            prediction_error=0.002 if is_correct else 0.015,
            realized_return=0.01 if is_correct else -0.01,
        ))

    # Current: accuracy collapsed to 30%
    cur_preds = []
    for i in range(40):
        is_correct = i < 12
        cur_preds.append(PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=50 - i),
            model_version="v1.0.0",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if is_correct else "INCORRECT",
            prediction_error=0.025,
            realized_return=-0.015,
        ))

    result = detector.evaluate_outcomes(ref_preds, cur_preds)
    assert result["status"] == str(DriftStatus.SEVERE_DRIFT)
    assert result["accuracy_drop"] >= 0.40


def test_regime_drift_detector() -> None:
    detector = RegimeDriftDetector(max_proportional_shift_threshold=0.20)

    # Ref: mostly TREND_UP
    ref_regimes = ["TREND_UP"] * 80 + ["SIDEWAYS"] * 20
    # Current: completely changed to HIGH_VOLATILITY
    cur_regimes = ["HIGH_VOLATILITY"] * 80 + ["SIDEWAYS"] * 20

    res = detector.evaluate_regimes(ref_regimes, cur_regimes)
    assert res["status"] in (str(DriftStatus.MODERATE_DRIFT), str(DriftStatus.SEVERE_DRIFT))
    assert res["max_regime_shift"] > 0.50
