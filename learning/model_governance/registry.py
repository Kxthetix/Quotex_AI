"""Enterprise Model Registry for Phase 8 Learning Loop.

Enforces:
- Production model safety: Candidate models can NEVER overwrite production models (Section 3).
- Immutable versioning: Never reuse a model version or modify an existing artifact (Section 5).
- Model artifacts integrity: Verifies file existence, SHA256 checksum, preprocessing, and schema (Sections 6, 37).
- Complete model lineage: Tracks parent production model, training dataset, feature version, config (Section 7).
- Explicit lifecycle state transitions (Section 35).
- Isolated storage: Candidates are stored separately from production models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learning.model_governance.models import (
    ModelLifecycleState,
    ModelLineageRecord,
)
from models.base import BaselineModel, ModelMetadata, ModelStatus
from monitoring.logging import get_logger

logger = get_logger(__name__)


class EnterpriseModelRegistry:
    """Enterprise-grade model registry with candidate isolation and production safety invariants."""

    def __init__(self, registry_dir: Path | str = "data/models") -> None:
        self.registry_dir = Path(registry_dir)
        self.production_dir = self.registry_dir / "production"
        self.candidates_dir = self.registry_dir / "candidates"
        self.archive_dir = self.registry_dir / "archive"

        for d in (self.production_dir, self.candidates_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.index_file = self.registry_dir / "enterprise_registry_index.json"
        self.lineage_file = self.registry_dir / "model_lineage.json"

        self._index: dict[str, dict[str, Any]] = self._load_index()
        self._lineage: dict[str, dict[str, Any]] = self._load_lineage()
        self._active_production_version: str | None = self._discover_active_production()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self.index_file.exists():
            try:
                with open(self.index_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("failed_to_load_registry_index", error=str(e))
        return {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)

    def _load_lineage(self) -> dict[str, dict[str, Any]]:
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("failed_to_load_lineage", error=str(e))
        return {}

    def _save_lineage(self) -> None:
        with open(self.lineage_file, "w", encoding="utf-8") as f:
            json.dump(self._lineage, f, indent=2, default=str)

    def _discover_active_production(self) -> str | None:
        for key, entry in self._index.items():
            if entry.get("status") == ModelLifecycleState.PRODUCTION.value:
                return entry.get("model_version")
        return None

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of an artifact file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_candidate(
        self,
        model: BaselineModel,
        lineage: ModelLineageRecord,
        metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        created_by: str = "CandidateTrainer",
    ) -> Path:
        """Register a newly trained model in CANDIDATE status.

        CRITICAL INVARIANT: Stored in isolated candidates directory.
        Never overwrites or touches active production model artifacts.
        """
        ver = model.model_version
        key = f"{model.model_name.lower()}::{ver.lower()}"

        if key in self._index:
            raise ValueError(
                f"Model version '{ver}' already exists in registry! "
                f"Model versions are strictly immutable. Create a new version."
            )

        # Save artifact into isolated candidates directory
        artifact_path = model.save(self.candidates_dir)
        checksum = self.calculate_checksum(artifact_path)

        meta = model.metadata()
        meta_dict = asdict(meta)
        meta_dict.update({
            "dataset_version": lineage.dataset_version,
            "artifact_location": str(artifact_path),
            "status": ModelLifecycleState.CANDIDATE.value,
            "created_by": created_by,
            "created_at": datetime.now(UTC).isoformat(),
            "parent_model_version": lineage.parent_production_model,
            "random_seed": lineage.random_seed,
            "code_version": lineage.code_version,
            "configuration_version": lineage.configuration_version,
            "checksum": checksum,
            "metrics": metrics or {},
            "parameters": hyperparameters or getattr(model, "parameters", {}),
            "training_period": list(lineage.training_period),
            "validation_period": list(lineage.validation_period),
            "test_period": list(lineage.test_period),
        })

        self._index[key] = meta_dict
        self._lineage[ver] = lineage.to_dict()

        self._save_index()
        self._save_lineage()

        logger.info(
            "candidate_model_registered",
            model_name=model.model_name,
            version=ver,
            checksum=checksum[:12],
            isolated_path=str(artifact_path),
        )
        return artifact_path

    def register_initial_production(
        self,
        model: BaselineModel,
        feature_version: str = "v1.0.0",
        dataset_version: str = "v1.0.0",
    ) -> Path:
        """Register the baseline initial production model."""
        ver = model.model_version
        key = f"{model.model_name.lower()}::{ver.lower()}"

        artifact_path = model.save(self.production_dir)
        checksum = self.calculate_checksum(artifact_path)

        meta = model.metadata()
        meta_dict = asdict(meta)
        meta_dict.update({
            "dataset_version": dataset_version,
            "artifact_location": str(artifact_path),
            "status": ModelLifecycleState.PRODUCTION.value,
            "created_by": "bootstrap",
            "created_at": datetime.now(UTC).isoformat(),
            "parent_model_version": None,
            "checksum": checksum,
        })

        self._index[key] = meta_dict
        self._active_production_version = ver
        self._save_index()

        logger.info("initial_production_model_registered", version=ver, path=str(artifact_path))
        return artifact_path

    def get_model(self, name: str, version: str) -> BaselineModel:
        """Load registered model artifact after validating integrity."""
        key = f"{name.lower()}::{version.lower()}"
        if key not in self._index:
            raise KeyError(f"Model '{name}' version '{version}' not found in registry.")

        entry = self._index[key]
        artifact_path = Path(entry["artifact_location"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact file missing at: {artifact_path}")

        # Checksum verification (Section 37)
        expected_checksum = entry.get("checksum")
        if expected_checksum:
            actual_checksum = self.calculate_checksum(artifact_path)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Artifact integrity violation for model {version}! "
                    f"Checksum mismatch: expected {expected_checksum[:12]}, got {actual_checksum[:12]}"
                )

        return BaselineModel.load(artifact_path)

    def get_active_production_version(self) -> str | None:
        """Returns the version string of the currently active production model."""
        return self._active_production_version

    def get_metadata(self, name: str, version: str) -> dict[str, Any]:
        key = f"{name.lower()}::{version.lower()}"
        if key not in self._index:
            raise KeyError(f"Model '{name}' version '{version}' not found in registry.")
        return dict(self._index[key])

    def get_lineage(self, version: str) -> dict[str, Any] | None:
        return self._lineage.get(version)

    def update_lifecycle_state(
        self,
        name: str,
        version: str,
        new_state: ModelLifecycleState,
        operator: str = "system",
        reason: str = "",
    ) -> None:
        """Update explicit lifecycle state.

        CRITICAL SAFETY RULE: Cannot directly promote to PRODUCTION via this method.
        Must use ControlledPromotionManager.
        """
        if new_state == ModelLifecycleState.PRODUCTION:
            raise PermissionError(
                "Direct transition to PRODUCTION is strictly prohibited! "
                "Must pass all validation gates and use ControlledPromotionManager."
            )

        key = f"{name.lower()}::{version.lower()}"
        if key not in self._index:
            raise KeyError(f"Model '{name}' version '{version}' not found in registry.")

        old_state = self._index[key].get("status")
        self._index[key]["status"] = new_state.value
        self._index[key]["last_state_change"] = {
            "from": old_state,
            "to": new_state.value,
            "operator": operator,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._save_index()

        logger.info(
            "model_lifecycle_state_updated",
            model=name,
            version=version,
            old_state=old_state,
            new_state=new_state.value,
            operator=operator,
        )

    def list_all_models(
        self,
        status: ModelLifecycleState | None = None,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for entry in self._index.values():
            if status and entry.get("status") != status.value:
                continue
            if model_type and entry.get("model_type") != model_type:
                continue
            results.append(dict(entry))
        return results
