# Model Rollback Operations

## 1. Overview & Objectives

If an approved and promoted model demonstrates unexpected degradation or anomalies in live execution, the `ControlledPromotionManager.rollback_model()` operation restores the previous approved production model.

## 2. Invariants & Safety Controls

1. **Known-Good Restoration**: Restores the previously verified, superseded production model.
2. **History Preservation**: The problematic model is marked `REJECTED`, **never deleted**.
3. **Artifact Integrity**: Checksum of restored model artifact is validated before activation.
4. **Auditable Record**: Generates an immutable `RollbackRecord` saved to `rollback_history.json`.
5. **No System Interruption**: Seamlessly re-points the active production version in `enterprise_registry_index.json`.
