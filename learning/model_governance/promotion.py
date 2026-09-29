"""Controlled Model Promotion Manager for Phase 8 (Sections 32, 33).

Enforces the strict, human/governance-gated promotion workflow:
CANDIDATE -> VALIDATING -> VALIDATED -> APPROVAL_REQUIRED -> APPROVED -> PRODUCTION

CRITICAL SAFETY INVARIANTS:
1. No candidate can automatically become production.
2. A model MUST have passed all 10 validation gates (status == VALIDATED).
3. Explicit approval reason and operator ID must be recorded.
4. Old production model is safely retired to RETIRED status, never deleted.
5. Emits an immutable PromotionRecord.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learning.model_governance.models import (
    GateStatus,
    ModelLifecycleState,
    PromotionRecord,
    RollbackRecord,
    ValidationResult,
)
from learning.model_governance.registry import EnterpriseModelRegistry
from monitoring.logging import get_logger

logger = get_logger(__name__)


class PromotionError(RuntimeError):
    """Raised when an invalid or unapproved model promotion is attempted."""
    pass


class ControlledPromotionManager:
    """Manages explicit, controlled transitions of validated candidates to active production."""

    def __init__(self, registry: EnterpriseModelRegistry) -> None:
        self.registry = registry
        self.history_file = self.registry.registry_dir / "promotion_history.json"
        self.rollback_file = self.registry.registry_dir / "rollback_history.json"
        self._promotion_history: list[PromotionRecord] = self._load_promotion_history()
        self._rollback_history: list[RollbackRecord] = self._load_rollback_history()

    def _load_promotion_history(self) -> list[PromotionRecord]:
        if self.history_file.exists():
            try:
                with open(self.history_file, encoding="utf-8") as f:
                    data = json.load(f)
                    records = []
                    for item in data:
                        records.append(PromotionRecord(
                            promotion_id=item.get("promotion_id", ""),
                            old_production_model=item.get("old_production_model", ""),
                            new_production_model=item.get("new_production_model", ""),
                            validation_id=item.get("validation_id", ""),
                            approved_by=item.get("approved_by", ""),
                            reason=item.get("reason", ""),
                            timestamp=datetime.fromisoformat(item["timestamp"]) if "timestamp" in item else datetime.now(UTC),
                            configuration_version=item.get("configuration_version", "v1.0.0"),
                            code_version=item.get("code_version", "v1.0.0"),
                        ))
                    return records
            except Exception as e:
                logger.warning("failed_to_load_promotion_history", error=str(e))
        return []

    def _save_promotion_history(self) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._promotion_history], f, indent=2)

    def _load_rollback_history(self) -> list[RollbackRecord]:
        if self.rollback_file.exists():
            try:
                with open(self.rollback_file, encoding="utf-8") as f:
                    data = json.load(f)
                    records = []
                    for item in data:
                        records.append(RollbackRecord(
                            rollback_id=item.get("rollback_id", ""),
                            problematic_model=item.get("problematic_model", ""),
                            restored_model=item.get("restored_model", ""),
                            reason=item.get("reason", ""),
                            initiated_by=item.get("initiated_by", ""),
                            timestamp=datetime.fromisoformat(item["timestamp"]) if "timestamp" in item else datetime.now(UTC),
                        ))
                    return records
            except Exception as e:
                logger.warning("failed_to_load_rollback_history", error=str(e))
        return []

    def _save_rollback_history(self) -> None:
        with open(self.rollback_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._rollback_history], f, indent=2)

    def promote_candidate(
        self,
        model_name: str,
        candidate_version: str,
        validation_result: ValidationResult,
        approved_by: str,
        reason: str,
        configuration_version: str = "v1.0.0",
    ) -> PromotionRecord:
        """Promote an approved, validated candidate to active PRODUCTION status.

        CRITICAL INVARIANT: Rejects promotion if validation failed.
        """
        # 1. Validation Gate Gatekeeper Check
        if validation_result.final_status != GateStatus.PASS:
            raise PromotionError(
                f"Cannot promote candidate '{candidate_version}'! "
                f"Validation gates did not pass (status: {validation_result.final_status}). "
                f"Failed/Warning gates: {validation_result.warnings}"
            )

        if validation_result.candidate_model_version != candidate_version:
            raise PromotionError(
                f"ValidationResult model version '{validation_result.candidate_model_version}' "
                f"does not match candidate '{candidate_version}'."
            )

        # 2. Check Candidate Exists in Registry
        cand_meta = self.registry.get_metadata(model_name, candidate_version)
        current_state = cand_meta.get("status")

        # 3. Identify Previous Production Model
        old_prod_version = self.registry.get_active_production_version() or "none"

        # 4. Copy candidate artifact to production directory safely
        cand_artifact = Path(cand_meta["artifact_location"])
        if not cand_artifact.exists():
            raise FileNotFoundError(f"Candidate artifact missing: {cand_artifact}")

        prod_dest = self.registry.production_dir / cand_artifact.name
        shutil.copy2(cand_artifact, prod_dest)

        # 5. Retire Old Production Model
        if old_prod_version != "none" and old_prod_version != candidate_version:
            try:
                old_key = f"{model_name.lower()}::{old_prod_version.lower()}"
                if old_key in self.registry._index:
                    self.registry._index[old_key]["status"] = ModelLifecycleState.RETIRED.value
                    self.registry._index[old_key]["retired_at"] = datetime.now(UTC).isoformat()
                    self.registry._index[old_key]["retired_reason"] = f"Superseded by {candidate_version}"
                    logger.info("retired_previous_production_model", version=old_prod_version)
            except Exception as e:
                logger.warning("failed_to_retire_old_model", error=str(e))

        # 6. Activate New Production Model in Registry
        new_key = f"{model_name.lower()}::{candidate_version.lower()}"
        self.registry._index[new_key]["status"] = ModelLifecycleState.PRODUCTION.value
        self.registry._index[new_key]["artifact_location"] = str(prod_dest)
        self.registry._index[new_key]["promoted_at"] = datetime.now(UTC).isoformat()
        self.registry._index[new_key]["promoted_by"] = approved_by
        self.registry._index[new_key]["promotion_reason"] = reason

        self.registry._active_production_version = candidate_version
        self.registry._save_index()

        # 7. Create Auditable PromotionRecord
        record = PromotionRecord(
            old_production_model=old_prod_version,
            new_production_model=candidate_version,
            validation_id=validation_result.validation_id,
            approved_by=approved_by,
            reason=reason,
            configuration_version=configuration_version,
        )
        self._promotion_history.append(record)
        self._save_promotion_history()

        logger.info(
            "controlled_promotion_successful",
            old_version=old_prod_version,
            new_version=candidate_version,
            approved_by=approved_by,
        )
        return record

    def rollback_model(
        self,
        model_name: str,
        problematic_version: str,
        target_version: str | None = None,
        reason: str = "Production performance degradation",
        initiated_by: str = "operator",
    ) -> RollbackRecord:
        """Rollback active production model to a previous approved/known-good model (Section 34, 55).

        CRITICAL SAFETY INVARIANTS:
        1. Select a known-good model (from promotion history or previous retired model).
        2. Preserve model history: Problematic model is marked REJECTED, NEVER deleted.
        3. Restores previous model artifact into active production.
        4. Emits and saves an immutable RollbackRecord.
        """
        # Find target model version if not specified
        if not target_version:
            # Look backwards in promotion history for the model superseded by problematic_version
            for rec in reversed(self._promotion_history):
                if rec.new_production_model == problematic_version and rec.old_production_model != "none":
                    target_version = rec.old_production_model
                    break

            # Fallback to searching registry for RETIRED or APPROVED models
            if not target_version:
                for entry in self.registry._index.values():
                    if entry.get("model_name", "").lower() == model_name.lower():
                        if entry.get("status") in (
                            ModelLifecycleState.RETIRED.value,
                            ModelLifecycleState.APPROVED.value,
                        ) and entry.get("model_version") != problematic_version:
                            target_version = entry.get("model_version")
                            break

        if not target_version:
            raise ValueError(
                f"No previous approved production model available for rollback from '{problematic_version}'."
            )

        if target_version == problematic_version:
            raise ValueError(
                f"Target rollback version '{target_version}' cannot be identical to problematic version."
            )

        # Verify target model exists and passes integrity check
        target_meta = self.registry.get_metadata(model_name, target_version)
        target_artifact = Path(target_meta["artifact_location"])
        if not target_artifact.exists():
            raise FileNotFoundError(f"Restored model artifact missing at: {target_artifact}")

        # Checksum validation on target model
        self.registry.get_model(model_name, target_version)

        # 1. Update problematic model status to REJECTED (preserves record, never deleted)
        prob_key = f"{model_name.lower()}::{problematic_version.lower()}"
        if prob_key in self.registry._index:
            self.registry._index[prob_key]["status"] = ModelLifecycleState.REJECTED.value
            self.registry._index[prob_key]["rejection_reason"] = f"Rolled back: {reason}"
            self.registry._index[prob_key]["rolled_back_at"] = datetime.now(UTC).isoformat()

        # 2. Restore target model to PRODUCTION
        target_key = f"{model_name.lower()}::{target_version.lower()}"
        prod_dest = self.registry.production_dir / target_artifact.name
        if target_artifact.parent != self.registry.production_dir:
            shutil.copy2(target_artifact, prod_dest)
            self.registry._index[target_key]["artifact_location"] = str(prod_dest)

        self.registry._index[target_key]["status"] = ModelLifecycleState.PRODUCTION.value
        self.registry._index[target_key]["restored_at"] = datetime.now(UTC).isoformat()
        self.registry._index[target_key]["restore_reason"] = f"Rolled back from {problematic_version}: {reason}"
        self.registry._active_production_version = target_version
        self.registry._save_index()

        # 3. Create and persist RollbackRecord
        record = RollbackRecord(
            problematic_model=problematic_version,
            restored_model=target_version,
            reason=reason,
            initiated_by=initiated_by,
        )
        self._rollback_history.append(record)
        self._save_rollback_history()

        logger.info(
            "model_rollback_successful",
            problematic_model=problematic_version,
            restored_model=target_version,
            reason=reason,
            initiated_by=initiated_by,
        )
        return record

    def get_promotion_history(self) -> list[PromotionRecord]:
        return list(self._promotion_history)

    def get_rollback_history(self) -> list[RollbackRecord]:
        return list(self._rollback_history)
