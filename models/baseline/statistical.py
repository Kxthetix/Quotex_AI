"""Baseline statistical models for Phase 4.

Provides transparent statistical reference benchmarks before machine learning:
1. MajorityClassModel (Majority class direction with historical frequency)
2. HistoricalMeanModel (Historical conditional mean / median)
3. RegimeConditionalModel (Historical outcome distribution conditioned on market regime)
"""

import numpy as np

from models.base import BaselineModel


class MajorityClassModel(BaselineModel):
    """Predicts the most frequent class from training data with historical empirical probability."""

    def __init__(
        self,
        model_name: str = "majority_class_baseline",
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        label_version: str = "v1.0.0",
        expected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            expected_features=expected_features,
            algorithm="MajorityClass",
        )
        self.model_type = "classifier"
        self.majority_class: int = 1
        self.class_probabilities: dict[int, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassModel":
        """Compute empirical class frequencies from training labels."""
        unique, counts = np.unique(y.astype(int), return_counts=True)
        total = len(y)
        self.class_probabilities = {
            int(u): float(c / total) for u, c in zip(unique, counts, strict=False)
        }

        max_idx = int(np.argmax(counts))
        self.majority_class = int(unique[max_idx])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict constant majority class."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.full(len(X), self.majority_class)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict empirical class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        prob_up = self.class_probabilities.get(1, 0.5)
        prob_down = self.class_probabilities.get(0, 0.5)
        # Returns [P(class 0), P(class 1)]
        row = np.array([prob_down, prob_up])
        return np.tile(row, (len(X), 1))


class HistoricalMeanModel(BaselineModel):
    """Predicts historical training mean and median for continuous regression targets."""

    def __init__(
        self,
        model_name: str = "historical_mean_baseline",
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        label_version: str = "v1.0.0",
        expected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            expected_features=expected_features,
            algorithm="HistoricalMean",
        )
        self.model_type = "regressor"
        self.mean_val: float = 0.0
        self.median_val: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistoricalMeanModel":
        """Compute mean and median of continuous training target."""
        clean_y = y[~np.isnan(y)]
        self.mean_val = float(np.mean(clean_y)) if len(clean_y) > 0 else 0.0
        self.median_val = float(np.median(clean_y)) if len(clean_y) > 0 else 0.0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict constant historical mean value."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.full(len(X), self.mean_val)


class RegimeConditionalModel(BaselineModel):
    """Statistical baseline predicting conditional expectation based on market regime."""

    def __init__(
        self,
        model_name: str = "regime_conditional_baseline",
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        label_version: str = "v1.0.0",
        expected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            expected_features=expected_features,
            algorithm="RegimeConditionalStatistics",
        )
        self.model_type = "regressor"
        self.conditional_means: dict[str, float] = {}
        self.global_mean: float = 0.0

    def fit(
        self, X: np.ndarray, y: np.ndarray, regimes: list[str] | None = None
    ) -> "RegimeConditionalModel":
        """Fit empirical mean conditioned on market regime."""
        self.global_mean = float(np.mean(y))
        if regimes is not None and len(regimes) == len(y):
            unique_r = set(regimes)
            for r in unique_r:
                indices = [i for i, reg in enumerate(regimes) if reg == r]
                subset_y = y[indices]
                self.conditional_means[r] = float(np.mean(subset_y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, regimes: list[str] | None = None) -> np.ndarray:
        """Predict conditional mean for each observation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        if regimes is None:
            return np.full(len(X), self.global_mean)
        return np.array([self.conditional_means.get(r, self.global_mean) for r in regimes])
