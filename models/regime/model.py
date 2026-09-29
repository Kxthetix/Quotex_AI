"""Learned regime classification model for Phase 4.

Predicts market regime using machine learning classifiers, kept strictly separate and
independently versioned from the rule-based MarketRegimeEngine.
"""

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from analysis.regime.models import MarketRegime
from models.base import BaselineModel


class LearnedRegimeModel(BaselineModel):
    """Machine-learning classifier for multi-class market regime forecasting."""

    def __init__(
        self,
        algorithm: str = "random_forest",
        model_name: str = "learned_regime_classifier",
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        label_version: str = "v1.0.0",
        expected_features: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            expected_features=expected_features,
            algorithm=algorithm,
        )
        self.model_type = "classifier"
        self.hyperparameters = hyperparameters or {}
        self.parameters = dict(self.hyperparameters)
        self.classes_: list[str] = [r.value for r in MarketRegime]
        self._estimator = self._init_estimator(algorithm, self.hyperparameters)

    def _init_estimator(self, algo: str, params: dict[str, Any]) -> Any:
        clean = algo.lower()
        if clean == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )
        elif clean in ("lightgbm", "lgbm"):
            return LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 5),
                random_state=params.get("random_state", 42),
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported algorithm for LearnedRegimeModel: '{algo}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedRegimeModel":
        """Train classifier on feature matrix X and string or integer regime targets y."""
        self._estimator.fit(X, y)
        if hasattr(self._estimator, "classes_"):
            self.classes_ = list(self._estimator.classes_)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict market regime category."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution across regimes."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self._estimator.predict_proba(X)
