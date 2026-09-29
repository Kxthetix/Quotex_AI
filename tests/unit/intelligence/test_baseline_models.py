"""Comprehensive tests for Phase 4 Baseline & Specialized ML Models.

Covers:
- Section 23 & 24: Baseline statistical models
- Section 26-31: Specialized ML models (Direction, Return, Volatility, TradeQuality, Regime)
- Section 32: Model registry foundation
- Section 40: Reproducibility & deterministic seeds
- Section 42: Strict feature/model compatibility checks
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.base import IncompatibleFeatureError, ModelStatus
from models.baseline.statistical import (
    HistoricalMeanModel,
    MajorityClassModel,
    RegimeConditionalModel,
)
from models.direction.model import DirectionModel
from models.regime.model import LearnedRegimeModel
from models.registry import LocalModelRegistry
from models.returns.model import ReturnModel
from models.trade_quality.model import TradeQualityModel
from models.volatility.model import VolatilityModel


@pytest.fixture
def synthetic_classification_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    # Slightly unbalanced binary targets
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def synthetic_regression_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = 0.05 * X[:, 0] - 0.02 * X[:, 1] + 0.01 * np.random.randn(60)
    return X, y


class TestStatisticalBaselines:
    """Tests for statistical baseline models."""

    def test_majority_class_model(self, synthetic_classification_data) -> None:
        X, y = synthetic_classification_data
        model = MajorityClassModel()
        model.fit(X, y)

        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(X)
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        # All predictions equal the computed majority class
        assert np.all(preds == model.majority_class)

    def test_historical_mean_model(self, synthetic_regression_data) -> None:
        X, y = synthetic_regression_data
        model = HistoricalMeanModel()
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert pytest.approx(preds[0]) == np.mean(y)

    def test_regime_conditional_model(self, synthetic_regression_data) -> None:
        X, y = synthetic_regression_data
        regimes = ["TREND_UP"] * 30 + ["SIDEWAYS"] * 30
        model = RegimeConditionalModel()
        model.fit(X, y, regimes=regimes)

        preds = model.predict(X, regimes=regimes)
        assert len(preds) == len(X)
        # Mean for TREND_UP
        expected_trend = float(np.mean(y[:30]))
        assert pytest.approx(preds[0]) == expected_trend


class TestSpecializedMLModels:
    """Tests for specialized models across algorithms."""

    @pytest.mark.parametrize("algo", ["logistic_regression", "random_forest", "lightgbm"])
    def test_direction_model(self, algo: str, synthetic_classification_data) -> None:
        X, y = synthetic_classification_data
        model = DirectionModel(algorithm=algo)
        model.fit(X, y)

        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(X)
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    @pytest.mark.parametrize("algo", ["ridge", "random_forest", "lightgbm"])
    def test_return_model(self, algo: str, synthetic_regression_data) -> None:
        X, y = synthetic_regression_data
        model = ReturnModel(algorithm=algo)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert isinstance(preds[0], (float, np.floating))

    @pytest.mark.parametrize("algo", ["ridge", "random_forest", "lightgbm"])
    def test_volatility_model_non_negative(self, algo: str, synthetic_regression_data) -> None:
        X, y = synthetic_regression_data
        y_vol = np.abs(y) + 0.01  # Strictly positive volatility targets
        model = VolatilityModel(algorithm=algo)
        model.fit(X, y_vol)

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(preds >= 0.0)

    @pytest.mark.parametrize("algo", ["logistic_regression", "random_forest", "lightgbm"])
    def test_trade_quality_model(self, algo: str, synthetic_classification_data) -> None:
        X, y = synthetic_classification_data
        model = TradeQualityModel(algorithm=algo)
        model.fit(X, y)

        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert len(preds) == len(X)
        assert probs.shape == (len(X), 2)

    @pytest.mark.parametrize("algo", ["random_forest", "lightgbm"])
    def test_learned_regime_model(self, algo: str, synthetic_classification_data) -> None:
        X, _ = synthetic_classification_data
        # Multi-class regime labels
        y_regime = np.array(["TREND_UP", "TREND_DOWN", "SIDEWAYS"] * 20)
        model = LearnedRegimeModel(algorithm=algo)
        model.fit(X, y_regime)

        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert len(preds) == len(X)
        assert probs.shape == (len(X), 3)


class TestFeatureCompatibilityAndGovernance:
    """Tests for feature compatibility checks, model persistence, and registry."""

    def test_feature_compatibility_validation(self) -> None:
        expected = ["feat1", "feat2", "feat3"]
        model = DirectionModel(
            algorithm="logistic_regression",
            feature_version="v1.0.0",
            expected_features=expected,
        )

        # 1. Matching features and version -> passes
        model.validate_features(["feat1", "feat2", "feat3"], "v1.0.0")

        # 2. Version mismatch -> raises IncompatibleFeatureError
        with pytest.raises(IncompatibleFeatureError, match="Feature version mismatch"):
            model.validate_features(["feat1", "feat2", "feat3"], "v2.0.0")

        # 3. Column mismatch -> raises IncompatibleFeatureError
        with pytest.raises(IncompatibleFeatureError, match="Feature schema mismatch"):
            model.validate_features(["feat1", "feat2", "feat_mismatched"], "v1.0.0")

    def test_model_save_load_and_registry(self, synthetic_classification_data) -> None:
        X, y = synthetic_classification_data
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = LocalModelRegistry(registry_dir=tmp_dir)

            model = DirectionModel(
                algorithm="logistic_regression",
                model_name="btc_direction_model",
                model_version="v1.0.0",
                expected_features=["f1", "f2", "f3", "f4"],
            )
            model.fit(X, y)

            # Register model with CANDIDATE status
            artifact_path = registry.register(
                model=model,
                dataset_id="ds_test_123",
                status=ModelStatus.CANDIDATE,
            )
            assert Path(artifact_path).exists()

            # Retrieve from registry
            loaded_model = registry.get("btc_direction_model", "v1.0.0")
            assert loaded_model.is_fitted
            assert loaded_model.model_name == "btc_direction_model"
            assert loaded_model.status == ModelStatus.CANDIDATE

            # Check predictions match loaded model
            np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))

            # Update status
            registry.update_status("btc_direction_model", "v1.0.0", ModelStatus.VALIDATION)
            models = registry.list_models(status=ModelStatus.VALIDATION)
            assert len(models) == 1
            assert models[0].status == ModelStatus.VALIDATION

    def test_reproducibility_deterministic_seed(self, synthetic_classification_data) -> None:
        """Verify that identical random seeds produce identical model predictions."""
        X, y = synthetic_classification_data

        model_1 = DirectionModel(
            algorithm="random_forest",
            hyperparameters={"random_state": 42, "n_estimators": 50},
        )
        model_1.fit(X, y)
        preds_1 = model_1.predict(X)

        model_2 = DirectionModel(
            algorithm="random_forest",
            hyperparameters={"random_state": 42, "n_estimators": 50},
        )
        model_2.fit(X, y)
        preds_2 = model_2.predict(X)

        np.testing.assert_array_equal(preds_1, preds_2)
