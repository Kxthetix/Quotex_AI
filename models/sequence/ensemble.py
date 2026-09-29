"""Multi-Model Ensemble Layer (Section 26).

Explicitly combines predictions from Baseline, Sequence, and Order-Book models
with auditable weights, versioning, and transparent provenance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaselineModel, ModelMetadata, ModelStatus


class EnsembleModel(BaselineModel):
    """Ensemble combiner unifying predictions across heterogeneous model architectures."""

    def __init__(
        self,
        sub_models: list[tuple[BaselineModel, float]],
        model_name: str = "production_ensemble",
        model_version: str = "ens_v1.0.0",
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            feature_version="multi_version",
            label_version="v1.0.0",
            algorithm="weighted_probability_consensus",
        )
        if not sub_models:
            raise ValueError("Ensemble must contain at least one sub-model.")

        total_weight = sum(w for _, w in sub_models)
        if total_weight <= 0:
            raise ValueError("Total ensemble weights must be positive.")

        # Normalize weights to sum to 1.0
        self.sub_models: list[tuple[BaselineModel, float]] = [
            (model, weight / total_weight) for model, weight in sub_models
        ]
        self.is_fitted = all(m.is_fitted for m, _ in self.sub_models)
        self.parameters = {
            "components": [
                {
                    "name": m.model_name,
                    "version": m.model_version,
                    "algorithm": m.algorithm,
                    "weight": round(w, 4),
                }
                for m, w in self.sub_models
            ]
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Fit any unfitted sub-models."""
        for model, _ in self.sub_models:
            if not model.is_fitted:
                model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted consensus class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("All sub-models in ensemble must be fitted.")

        composite_probs: np.ndarray | None = None

        for model, weight in self.sub_models:
            probs = model.predict_proba(X)
            if probs is None:
                # Fall back to 1-hot from predict()
                preds = model.predict(X)
                probs = np.zeros((len(preds), 2), dtype=np.float32)
                probs[np.arange(len(preds)), preds.astype(int)] = 1.0

            if composite_probs is None:
                composite_probs = weight * probs
            else:
                composite_probs += weight * probs

        return composite_probs  # type: ignore[return-value]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class with highest weighted ensemble probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
