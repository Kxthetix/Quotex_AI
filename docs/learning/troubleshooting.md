# Learning Loop Troubleshooting Guide

## Common Issues & Diagnostics

### 1. `PromotionError: Cannot promote candidate ... Validation gates did not pass`
- **Cause**: Candidate model attempted to be promoted with `final_status != PASS`.
- **Resolution**: Inspect `ValidationResult.warnings` to identify which of the 10 gates failed (e.g. `GATE_8_DRIFT_ROBUSTNESS` or `GATE_9_RISK_CONSTRAINTS`). Ensure model meets all threshold criteria.

### 2. `PermissionError: Direct transition to PRODUCTION is strictly prohibited!`
- **Cause**: Code called `registry.update_lifecycle_state(..., ModelLifecycleState.PRODUCTION)`.
- **Resolution**: Direct transition is an invariant violation. Use `ControlledPromotionManager.promote_candidate()`.

### 3. `ValueError: Artifact integrity violation for model ... Checksum mismatch`
- **Cause**: Model file on disk was modified, corrupted, or replaced outside the registry.
- **Resolution**: Do not edit model files directly. If an artifact was corrupted, restore from backup or retrain a new version.

### 4. `Sample size ... is below minimum requirement`
- **Cause**: Model auditor evaluated fewer predictions than `min_audit_sample_size` (default 30).
- **Resolution**: Model status is `INSUFFICIENT_DATA`. Allow the paper-trading session to collect more evaluated observations.

### 5. `Feature version mismatch or schema incompatibility`
- **Cause**: Model feature version does not match active feature engine schema.
- **Resolution**: Train a new candidate using the current feature engine version.
