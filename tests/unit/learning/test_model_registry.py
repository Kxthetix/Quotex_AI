"""Tests for Enterprise Model Registry (Phase 8 Sections 4, 5, 6, 7, 35, 37)."""

from datetime import UTC, datetime
from pathlib import Path
import pytest

from learning.model_governance.models import (
    ModelLifecycleState,
    ModelLineageRecord,
)
from learning.model_governance.registry import EnterpriseModelRegistry
from models.direction.model import DirectionModel


@pytest.fixture
def temp_registry_dir(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "models_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    return reg_dir


@pytest.fixture
def mock_direction_model() -> DirectionModel:
    model = DirectionModel(
        algorithm="random_forest",
        model_name="direction_classifier",
        model_version="dir_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        expected_features=["feat_a", "feat_b"],
        hyperparameters={"n_estimators": 10},
    )
    return model


def test_register_initial_production(temp_registry_dir: Path, mock_direction_model: DirectionModel) -> None:
    registry = EnterpriseModelRegistry(temp_registry_dir)
    artifact_path = registry.register_initial_production(
        model=mock_direction_model,
        feature_version="v1.0.0",
        dataset_version="ds_v001",
    )

    assert artifact_path.exists()
    assert registry.get_active_production_version() == "dir_v001"
    meta = registry.get_metadata("direction_classifier", "dir_v001")
    assert meta["status"] == ModelLifecycleState.PRODUCTION.value
    assert "checksum" in meta


def test_register_candidate_isolated(temp_registry_dir: Path, mock_direction_model: DirectionModel) -> None:
    registry = EnterpriseModelRegistry(temp_registry_dir)
    # Register prod
    registry.register_initial_production(mock_direction_model)

    # Create candidate
    cand_model = DirectionModel(
        algorithm="random_forest",
        model_name="direction_classifier",
        model_version="dir_cand_v002",
        expected_features=["feat_a", "feat_b"],
    )
    lineage = ModelLineageRecord(
        model_version="dir_cand_v002",
        parent_production_model="dir_v001",
        dataset_version="ds_v002",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("2026-01-01", "2026-06-01"),
        validation_period=("2026-06-01", "2026-07-01"),
        test_period=("2026-07-01", "2026-08-01"),
    )

    cand_path = registry.register_candidate(
        model=cand_model,
        lineage=lineage,
        metrics={"val_acc": 0.55},
    )

    assert cand_path.exists()
    assert "candidates" in str(cand_path)
    # Production model remains strictly unchanged
    assert registry.get_active_production_version() == "dir_v001"

    cand_meta = registry.get_metadata("direction_classifier", "dir_cand_v002")
    assert cand_meta["status"] == ModelLifecycleState.CANDIDATE.value
    assert cand_meta["parent_model_version"] == "dir_v001"

    # Lineage retrieval
    saved_lineage = registry.get_lineage("dir_cand_v002")
    assert saved_lineage is not None
    assert saved_lineage["parent_production_model"] == "dir_v001"


def test_immutable_version_protection(temp_registry_dir: Path, mock_direction_model: DirectionModel) -> None:
    registry = EnterpriseModelRegistry(temp_registry_dir)
    registry.register_initial_production(mock_direction_model)

    lineage = ModelLineageRecord(
        model_version="dir_v001",  # Same version as production
        parent_production_model=None,
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )

    # Attempting to register existing version as candidate must fail
    with pytest.raises(ValueError, match="already exists"):
        registry.register_candidate(mock_direction_model, lineage)


def test_artifact_integrity_verification(temp_registry_dir: Path, mock_direction_model: DirectionModel) -> None:
    registry = EnterpriseModelRegistry(temp_registry_dir)
    artifact_path = registry.register_initial_production(mock_direction_model)

    # Valid load passes
    loaded_model = registry.get_model("direction_classifier", "dir_v001")
    assert loaded_model.model_version == "dir_v001"

    # Corrupt the file
    with open(artifact_path, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    # Loading corrupted artifact must raise ValueError due to checksum mismatch
    with pytest.raises(ValueError, match="Artifact integrity violation"):
        registry.get_model("direction_classifier", "dir_v001")


def test_lifecycle_state_transition_safety(temp_registry_dir: Path, mock_direction_model: DirectionModel) -> None:
    registry = EnterpriseModelRegistry(temp_registry_dir)
    cand_model = DirectionModel(
        model_name="direction_classifier",
        model_version="dir_cand_v003",
    )
    lineage = ModelLineageRecord(
        model_version="dir_cand_v003",
        parent_production_model=None,
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry.register_candidate(cand_model, lineage)

    # Can transition to VALIDATING and VALIDATED
    registry.update_lifecycle_state("direction_classifier", "dir_cand_v003", ModelLifecycleState.VALIDATING)
    meta = registry.get_metadata("direction_classifier", "dir_cand_v003")
    assert meta["status"] == ModelLifecycleState.VALIDATING.value

    # Direct transition to PRODUCTION is strictly prohibited
    with pytest.raises(PermissionError, match="strictly prohibited"):
        registry.update_lifecycle_state("direction_classifier", "dir_cand_v003", ModelLifecycleState.PRODUCTION)
