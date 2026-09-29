"""Return magnitude regression model for Phase 4.

Predicts continuous forward price return magnitude over a defined horizon.
Invariant: Pure regression estimate; does not imply guaranteed returns.
"""

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from models.base import BaselineModel


class ReturnModel(BaselineModel):
    """Forward return regressor supporting Ridge, Random Forest, and LightGBM."""

    def __init__(
        self,
        algorithm: str = "ridge",
        model_name: str = "return_regressor",
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
        self.model_type = "regressor"
        self.hyperparameters = hyperparameters or {}
        self.parameters = dict(self.hyperparameters)
        self._estimator = self._init_estimator(algorithm, self.hyperparameters)

    def _init_estimator(self, algo: str, params: dict[str, Any]) -> Any:
        clean_algo = algo.lower()
        if clean_algo == "ridge":
            return Ridge(
                alpha=params.get("alpha", 1.0),
                random_state=params.get("random_state", 42),
            )
        elif clean_algo == "random_forest":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 5),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )
        elif clean_algo in ("lightgbm", "lgbm"):
            return LGBMRegressor(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 5),
                random_state=params.get("random_state", 42),
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported algorithm for ReturnModel: '{algo}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReturnModel":
        """Train regressor on feature matrix X and forward return targets y."""
        clean_y = y.astype(float)
        self._estimator.fit(X, clean_y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expected forward percentage return."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._estimator.predict(X)
