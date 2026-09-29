"""Advanced Temporal Sequence Model (Sections 19, 20, 22).

Implements TemporalSequenceModel conforming to the canonical BaselineModel interface.
Supports 3D temporal sequence inputs (N, Sequence_Length, Features) with
temporal convolution and exponential recency pooling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.base import BaselineModel, ModelMetadata, ModelStatus


class TemporalSequenceModel(BaselineModel):
    """Temporal Sequence Model processing rolling temporal feature sequences (Section 19, 20)."""

    def __init__(
        self,
        model_name: str = "temporal_sequence_model",
        model_version: str = "seq_v1.0.0",
        sequence_length: int = 15,
        decay_factor: float = 0.95,
        expected_features: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version="orderbook_features_v1",
            label_version="v1.0.0",
            expected_features=expected_features,
            algorithm="temporal_convolution_recency_pooled",
        )
        self.sequence_length = sequence_length
        self.decay_factor = decay_factor
        self.random_seed = random_seed
        self._clf = LogisticRegression(random_state=random_seed, max_iter=200)
        self.parameters = {
            "sequence_length": sequence_length,
            "decay_factor": decay_factor,
            "random_seed": random_seed,
            "algorithm": self.algorithm,
        }

    def _prepare_sequence_features(self, X: np.ndarray) -> np.ndarray:
        """Transform (N, L, D) 3D sequence array into temporally weighted 2D representations.
        
        Applies exponential recency weighting and delta velocity across sequence steps.
        If input is already 2D (N, D), processes directly.
        """
        if X.ndim == 2:
            return X

        if X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got ndim={X.ndim}")

        N, L, D = X.shape
        # Weight vector emphasizing recent steps: [decay^(L-1), ..., decay^1, 1.0]
        weights = np.array([self.decay_factor ** (L - 1 - t) for t in range(L)], dtype=np.float32)
        weights /= weights.sum()

        # 1. Temporally weighted average across time steps: (N, D)
        weighted_mean = np.sum(X * weights.reshape(1, L, 1), axis=1)

        # 2. Sequence velocity: delta between last and first time step: (N, D)
        velocity = X[:, -1, :] - X[:, 0, :]

        # 3. Concatenate summary representations: (N, 2*D)
        pooled = np.concatenate([weighted_mean, velocity], axis=1)
        return pooled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalSequenceModel":
        """Fit model to training sequences X and target vector y."""
        X_prep = self._prepare_sequence_features(X)
        self._clf.fit(X_prep, y)
        self.is_fitted = True
        self.status = ModelStatus.VALIDATED
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary direction classes (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X_prep = self._prepare_sequence_features(X)
        return self._clf.predict(X_prep)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        X_prep = self._prepare_sequence_features(X)
        return self._clf.predict_proba(X_prep)
