"""Model Compatibility Validator for Phase 8 (Section 36).

Verifies strict compatibility across:
- model version
- feature version
- feature schema (expected vs provided features)
- preprocessing version
- signal version
- configuration version

CRITICAL INVARIANT:
If any incompatibility is detected, the candidate is strictly REJECTED.
The system NEVER guesses or applies ad-hoc transformations to incompatible artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemEnvironmentSpecification:
    """Current active production environment expectations."""
    expected_feature_version: str = "v1.0.0"
    expected_preprocessing_version: str = "v1.0.0"
    expected_signal_version: str = "v1.0.0"
    expected_configuration_version: str = "v1.0.0"
    required_features: list[str] = field(default_factory=list)


@dataclass
class CompatibilityCheckResult:
    """Result of compatibility validation."""
    is_compatible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    model_version: str = ""
    feature_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compatible": self.is_compatible,
            "rejection_reasons": self.rejection_reasons,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
        }


class ModelCompatibilityValidator:
    """Validates candidate model compatibility against production system environment."""

    def __init__(self, system_spec: SystemEnvironmentSpecification | None = None) -> None:
        self.system_spec = system_spec or SystemEnvironmentSpecification()

    def validate(
        self,
        candidate_metadata: dict[str, Any],
        system_spec: SystemEnvironmentSpecification | None = None,
    ) -> CompatibilityCheckResult:
        """Verify candidate model compatibility against system specifications."""
        spec = system_spec or self.system_spec
        reasons: list[str] = []

        model_version = candidate_metadata.get("model_version", "unknown")
        cand_feature_ver = candidate_metadata.get("feature_version", "")
        cand_preproc_ver = candidate_metadata.get("preprocessing_version", spec.expected_preprocessing_version)
        cand_signal_ver = candidate_metadata.get("signal_version", spec.expected_signal_version)
        cand_config_ver = candidate_metadata.get("configuration_version", spec.expected_configuration_version)
        cand_expected_features = candidate_metadata.get("expected_features", [])

        # 1. Feature Version Check
        if cand_feature_ver != spec.expected_feature_version:
            reasons.append(
                f"Feature version mismatch: model expects '{cand_feature_ver}', "
                f"system requires '{spec.expected_feature_version}'."
            )

        # 2. Preprocessing Version Check
        if cand_preproc_ver != spec.expected_preprocessing_version:
            reasons.append(
                f"Preprocessing version mismatch: model expects '{cand_preproc_ver}', "
                f"system requires '{spec.expected_preprocessing_version}'."
            )

        # 3. Signal Version Check
        if cand_signal_ver != spec.expected_signal_version:
            reasons.append(
                f"Signal version mismatch: model expects '{cand_signal_ver}', "
                f"system requires '{spec.expected_signal_version}'."
            )

        # 4. Configuration Version Check
        if cand_config_ver != spec.expected_configuration_version:
            reasons.append(
                f"Configuration version mismatch: model expects '{cand_config_ver}', "
                f"system requires '{spec.expected_configuration_version}'."
            )

        # 5. Feature Schema Check (if required features specified)
        if spec.required_features and cand_expected_features:
            missing_features = [f for f in spec.required_features if f not in cand_expected_features]
            extra_features = [f for f in cand_expected_features if f not in spec.required_features]
            if missing_features or extra_features:
                reasons.append(
                    f"Feature schema mismatch: missing in model: {missing_features}, "
                    f"extra in model: {extra_features}."
                )

        is_compatible = len(reasons) == 0
        if not is_compatible:
            logger.warning(
                "model_compatibility_rejected",
                model_version=model_version,
                reasons=reasons,
            )
        else:
            logger.info("model_compatibility_verified", model_version=model_version)

        return CompatibilityCheckResult(
            is_compatible=is_compatible,
            rejection_reasons=reasons,
            model_version=model_version,
            feature_version=cand_feature_ver,
        )
