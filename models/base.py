"""Base model abstractions, metadata, and compatibility validation for Phase 4.

Defines the standard BaselineModel interface and strict feature-compatibility enforcement.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelStatus(StrEnum):
    """Lifecycle status of a registered model."""

    DEVELOPMENT = "DEVELOPMENT"
    EXPERIMENT = "EXPERIMENT"  # Backward compatibility for Phase 4
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    VALIDATION = "VALIDATION"  # Backward compatibility for Phase 4
    VALIDATED = "VALIDATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelMetadata:
    """Comprehensive provenance and governance metadata for a model artifact."""

    model_id: str
    model_name: str
    model_version: str
    model_type: str  # "classifier", "regressor"
    algorithm: str
    feature_version: str
    label_version: str
    expected_features: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    training_period: tuple[str, str] = ("", "")
    validation_period: tuple[str, str] = ("", "")
    test_period: tuple[str, str] = ("", "")
    dataset_version: str = "v1.0.0"
    status: ModelStatus = ModelStatus.EXPERIMENT
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    created_by: str = "system"
    parent_model_version: str | None = None
    random_seed: int = 42
    code_version: str = "v1.0.0"
    configuration_version: str = "v1.0.0"
    checksum: str = ""


class IncompatibleFeatureError(ValueError):
    """Raised when an inference request supplies an incompatible feature schema or version."""

    pass


class BaselineModel(ABC):
    """Standard model interface for all Phase 4 statistical baselines and machine learning models."""

    def __init__(
        self,
        model_name: str,
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        label_version: str = "v1.0.0",
        expected_features: list[str] | None = None,
        algorithm: str = "custom",
    ) -> None:
        self.model_id = f"model_{uuid.uuid4().hex[:12]}"
        self.model_name = model_name
        self.model_version = model_version
        self.feature_version = feature_version
        self.label_version = label_version
        self.expected_features = expected_features or []
        self.algorithm = algorithm
        self.is_fitted: bool = False
        self.metrics: dict[str, float] = {}
        self.parameters: dict[str, Any] = {}
        self.training_period: tuple[str, str] = ("", "")
        self.status: ModelStatus = ModelStatus.EXPERIMENT

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """Fit model to training feature matrix X and target vector y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Generate class probability predictions where applicable."""
        return None

    def validate_features(self, feature_names: list[str], feature_version: str) -> None:
        """Strictly validate feature version and column ordering before inference.

        Raises IncompatibleFeatureError if feature schema differs from model expectations.
        """
        if feature_version != self.feature_version:
            raise IncompatibleFeatureError(
                f"Feature version mismatch for model '{self.model_name}': "
                f"Model expects '{self.feature_version}', received '{feature_version}'."
            )
        if self.expected_features and feature_names != self.expected_features:
            missing = set(self.expected_features) - set(feature_names)
            unexpected = set(feature_names) - set(self.expected_features)
            raise IncompatibleFeatureError(
                f"Feature schema mismatch for model '{self.model_name}'. "
                f"Missing features: {sorted(missing)}, Unexpected features: {sorted(unexpected)}."
            )

    def metadata(self) -> ModelMetadata:
        """Export model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.model_version,
            model_type=getattr(self, "model_type", "custom"),
            algorithm=self.algorithm,
            feature_version=self.feature_version,
            label_version=self.label_version,
            expected_features=list(self.expected_features),
            parameters=dict(self.parameters),
            metrics=dict(self.metrics),
            training_period=self.training_period,
            status=self.status,
            created_at=datetime.now(UTC).isoformat(),
        )

    def save(self, directory: Path | str) -> Path:
        """Persist model artifact alongside its governance metadata."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = target_dir / f"{self.model_name}_{self.model_version}.joblib"
        metadata_path = target_dir / f"{self.model_name}_{self.model_version}.json"

        joblib.dump(self, artifact_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata()), f, indent=2)

        return artifact_path

    @classmethod
    def load(cls, artifact_path: Path | str) -> "BaselineModel":
        """Load a persisted model artifact."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        model = joblib.load(path)
        if not isinstance(model, BaselineModel):
            raise TypeError(f"Loaded object is not a BaselineModel: {type(model)}")
        return model
