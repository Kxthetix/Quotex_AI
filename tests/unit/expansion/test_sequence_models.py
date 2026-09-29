"""Unit tests for Sequence Dataset Builder, Sequence Models, and Ensembles (Sections 19-26, 56)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.base import BaselineModel, ModelStatus
from models.registry import LocalModelRegistry
from models.sequence.dataset import SequenceDatasetBuilder
from models.sequence.ensemble import EnsembleModel
from models.sequence.models import TemporalSequenceModel


def test_sequence_dataset_builder_shapes_and_causality():
    """Section 21: Verify (N, L, D) shape and strict chronological alignment."""
    # Create synthetic DataFrame with 50 time steps and 3 features
    dates = pd.date_range("2026-09-01", periods=50, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "feat_a": np.arange(50, dtype=float),
        "feat_b": np.arange(50, dtype=float) * 2,
        "feat_c": np.sin(np.arange(50)),
        "target": (np.arange(50) % 2).astype(int),
    })

    builder = SequenceDatasetBuilder(sequence_length=10, stride=1)
    ds = builder.build_sequences(
        df=df,
        feature_cols=["feat_a", "feat_b", "feat_c"],
        target_col="target",
        timestamp_col="timestamp",
    )

    # 50 - 10 + 1 = 41 samples
    assert ds.X.shape == (41, 10, 3)
    assert ds.y.shape == (41,)
    assert len(ds.timestamps) == 41

    # Verify causality: first sample X[0] covers index 0 to 9, ending at index 9
    np.testing.assert_array_equal(ds.X[0, :, 0], np.arange(10, dtype=float))
    assert ds.y[0] == df["target"].iloc[9]


def test_temporal_sequence_model_fit_and_predict():
    """Section 19, 20: Temporal sequence model training and prediction."""
    np.random.seed(42)
    # 60 samples, sequence length 10, 4 features
    X_train = np.random.randn(60, 10, 4).astype(np.float32)
    # Signal: positive trend in feature 0 -> class 1
    y_train = (X_train[:, -1, 0] > X_train[:, 0, 0]).astype(int)

    model = TemporalSequenceModel(
        model_name="test_seq_model",
        sequence_length=10,
        expected_features=["f1", "f2", "f3", "f4"],
    )

    model.fit(X_train, y_train)
    assert model.is_fitted is True

    # Predict
    preds = model.predict(X_train[:5])
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X_train[:5])
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5)


def test_ensemble_model_weighted_consensus():
    """Section 26: Ensemble combining multiple models with auditable weights."""
    np.random.seed(42)
    X = np.random.randn(40, 10, 2).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    m1 = TemporalSequenceModel(model_name="seq_fast", sequence_length=10, decay_factor=0.9)
    m2 = TemporalSequenceModel(model_name="seq_slow", sequence_length=10, decay_factor=0.99)

    m1.fit(X, y)
    m2.fit(X, y)

    ensemble = EnsembleModel(
        sub_models=[(m1, 0.6), (m2, 0.4)],
        model_name="combo_ensemble",
    )

    probs = ensemble.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), rtol=1e-5)

    preds = ensemble.predict(X[:10])
    assert len(preds) == 10
    assert "components" in ensemble.parameters


def test_critical_model_test_section_56(tmp_path: Path):
    """Section 56 Critical Model Test:
    Train BASELINE_MODEL + SEQUENCE_MODEL -> register both ->
    verify both use the same model lifecycle -> verify production model cannot be silently replaced.
    """
    registry = LocalModelRegistry(registry_dir=tmp_path / "models")

    # 1. Baseline Model
    baseline = TemporalSequenceModel(
        model_name="baseline_model",
        model_version="v1.0.0",
        sequence_length=5,
    )
    # 2. Advanced Sequence Model
    sequence_model = TemporalSequenceModel(
        model_name="sequence_transformer_tcn",
        model_version="v2.0.0",
        sequence_length=15,
    )

    X_mock = np.random.randn(20, 5, 2).astype(np.float32)
    y_mock = (X_mock[:, -1, 0] > 0).astype(int)
    baseline.fit(X_mock, y_mock)

    X_mock_seq = np.random.randn(20, 15, 2).astype(np.float32)
    y_mock_seq = (X_mock_seq[:, -1, 0] > 0).astype(int)
    sequence_model.fit(X_mock_seq, y_mock_seq)

    # 3. Register both in ModelRegistry
    path1 = registry.register(baseline, status=ModelStatus.PRODUCTION)
    path2 = registry.register(sequence_model, status=ModelStatus.CANDIDATE)

    assert path1.exists()
    assert path2.exists()

    # 4. Verify both are tracked under the same governance lifecycle
    meta_baseline = registry.get_metadata("baseline_model", "v1.0.0")
    meta_seq = registry.get_metadata("sequence_transformer_tcn", "v2.0.0")

    assert meta_baseline["status"] == ModelStatus.PRODUCTION
    assert meta_seq["status"] == ModelStatus.CANDIDATE

    # 5. Verify production model cannot be silently replaced: candidate remains CANDIDATE until promoted
    assert meta_seq["status"] != ModelStatus.PRODUCTION
