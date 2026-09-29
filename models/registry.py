"""Local Model Registry foundation for Phase 4.

Provides auditable artifact storage, metadata tracking, and governance status management
(EXPERIMENT, CANDIDATE, VALIDATION, REJECTED) without automatic production promotion.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from models.base import BaselineModel, ModelMetadata, ModelStatus
from monitoring.logging import get_logger

logger = get_logger(__name__)


class LocalModelRegistry:
    """Local, auditable repository for model artifacts and governance metadata."""

    def __init__(self, registry_dir: Path | str = "data/models") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "registry_index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        if self.index_file.exists():
            try:
                with open(self.index_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to parse model registry index: %s", e)
        return {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def _make_key(self, name: str, version: str) -> str:
        return f"{name.lower()}::{version.lower()}"

    def register(
        self,
        model: BaselineModel,
        dataset_id: str = "",
        status: ModelStatus = ModelStatus.EXPERIMENT,
    ) -> Path:
        """Save a model artifact and register its metadata in the registry."""
        model.status = status
        artifact_path = model.save(self.registry_dir)

        meta = model.metadata()
        meta_dict = asdict(meta)
        meta_dict["dataset_id"] = dataset_id
        meta_dict["artifact_path"] = str(artifact_path)
        meta_dict["registered_at"] = datetime.now(UTC).isoformat()

        key = self._make_key(model.model_name, model.model_version)
        self._index[key] = meta_dict
        self._save_index()

        logger.info(
            "Registered model '%s' version '%s' [Status: %s]",
            model.model_name,
            model.model_version,
            status.value,
        )
        return artifact_path

    def get(self, name: str, version: str = "v1.0.0") -> BaselineModel:
        """Load a registered model artifact."""
        key = self._make_key(name, version)
        if key not in self._index:
            raise KeyError(f"Model '{name}' with version '{version}' not found in registry.")

        entry = self._index[key]
        artifact_path = Path(entry["artifact_path"])
        return BaselineModel.load(artifact_path)

    def get_metadata(self, name: str, version: str = "v1.0.0") -> dict:
        """Retrieve registered metadata dictionary for a model."""
        key = self._make_key(name, version)
        if key not in self._index:
            raise KeyError(f"Model '{name}' with version '{version}' not found in registry.")
        return self._index[key]

    def list_models(
        self,
        model_type: str | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelMetadata]:
        """List registered model metadata matching optional filters."""
        results: list[ModelMetadata] = []
        for entry in self._index.values():
            if model_type and entry.get("model_type") != model_type:
                continue
            if status and entry.get("status") != status.value:
                continue

            meta = ModelMetadata(
                model_id=entry["model_id"],
                model_name=entry["model_name"],
                model_version=entry["model_version"],
                model_type=entry["model_type"],
                algorithm=entry["algorithm"],
                feature_version=entry["feature_version"],
                label_version=entry["label_version"],
                expected_features=entry["expected_features"],
                parameters=entry["parameters"],
                metrics=entry["metrics"],
                training_period=tuple(entry.get("training_period", ("", ""))),
                status=ModelStatus(entry.get("status", "EXPERIMENT")),
                created_at=entry.get("created_at", ""),
            )
            results.append(meta)
        return results

    def update_status(self, name: str, version: str, new_status: ModelStatus) -> None:
        """Update governance status of a registered model."""
        key = self._make_key(name, version)
        if key not in self._index:
            raise KeyError(f"Model '{name}' version '{version}' not found in registry.")
        self._index[key]["status"] = new_status.value
        self._save_index()
        logger.info("Updated status of '%s' [%s] to %s", name, version, new_status.value)
