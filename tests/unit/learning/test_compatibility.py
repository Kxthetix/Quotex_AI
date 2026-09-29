"""Tests for Model Compatibility Validator (Phase 8 Section 36)."""

import pytest

from learning.model_governance.compatibility import (
    ModelCompatibilityValidator,
    SystemEnvironmentSpecification,
)


def test_compatibility_validation() -> None:
    system_spec = SystemEnvironmentSpecification(
        expected_feature_version="v1.0.0",
        expected_preprocessing_version="v1.0.0",
        expected_signal_version="v1.0.0",
        expected_configuration_version="v1.0.0",
        required_features=["rsi_14", "macd", "volatility_20"],
    )
    validator = ModelCompatibilityValidator(system_spec)

    # 1. Fully Compatible Candidate
    valid_meta = {
        "model_version": "cand_v001",
        "feature_version": "v1.0.0",
        "preprocessing_version": "v1.0.0",
        "signal_version": "v1.0.0",
        "configuration_version": "v1.0.0",
        "expected_features": ["rsi_14", "macd", "volatility_20"],
    }
    res_valid = validator.validate(valid_meta)
    assert res_valid.is_compatible
    assert len(res_valid.rejection_reasons) == 0

    # 2. Incompatible Feature Version
    bad_feat_meta = dict(valid_meta)
    bad_feat_meta["feature_version"] = "v2.0.0"
    res_bad_feat = validator.validate(bad_feat_meta)
    assert not res_bad_feat.is_compatible
    assert any("Feature version mismatch" in r for r in res_bad_feat.rejection_reasons)

    # 3. Schema Incompatibility (missing feature)
    bad_schema_meta = dict(valid_meta)
    bad_schema_meta["expected_features"] = ["rsi_14", "macd"]  # missing volatility_20
    res_bad_schema = validator.validate(bad_schema_meta)
    assert not res_bad_schema.is_compatible
    assert any("Feature schema mismatch" in r for r in res_bad_schema.rejection_reasons)

    # 4. Incompatible Signal Version
    bad_signal_meta = dict(valid_meta)
    bad_signal_meta["signal_version"] = "v0.9.0"
    res_bad_signal = validator.validate(bad_signal_meta)
    assert not res_bad_signal.is_compatible
    assert any("Signal version mismatch" in r for r in res_bad_signal.rejection_reasons)
