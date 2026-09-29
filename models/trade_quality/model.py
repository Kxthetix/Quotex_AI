"""Trade-quality classification model for Phase 4.

Predicts the probability that price reaches a predefined target before an invalidation stop within horizon H.
Invariant: Output is a model-estimated probability; does not constitute a trading decision.
"""

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models.base import BaselineModel


class TradeQualityModel(BaselineModel):
    """Trade quality classifier supporting Logistic Regression, Random Forest, and LightGBM."""

    def __init__(
        self,
        algorithm: str = "random_forest",
        model_name: str = "trade_quality_classifier",
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
        self._estimator = self._init_estimator(algorithm, self.hyperparameters)

    def _init_estimator(self, algo: str, params: dict[str, Any]) -> Any:
        clean_algo = algo.lower()
        if clean_algo == "logistic_regression":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                random_state=params.get("random_state", 42),
            )
        elif clean_algo == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 5),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )
        elif clean_algo in ("lightgbm", "lgbm"):
            return LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 5),
                random_state=params.get("random_state", 42),
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported algorithm for TradeQualityModel: '{algo}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TradeQualityModel":
        """Train classifier on feature matrix X and binary trade-quality labels y."""
        clean_y = y.astype(int)
        self._estimator.fit(X, clean_y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outcome (1 = target hit first, 0 = stop hit first/timeout)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distribution [P(Failed), P(TargetHit)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self._estimator.predict_proba(X)
