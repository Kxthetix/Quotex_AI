# Advanced Model Validation & Registry Integration

## 1. 5 Validation Gates

Every candidate sequence model must pass the Phase 8 validation gates before promotion:
1. **Performance Gate**: Out-of-sample directional accuracy and Brier score exceed baseline thresholds.
2. **Stability Gate**: Prediction variance across regimes does not exhibit sudden degradation.
3. **Calibration Gate**: Expected Calibration Error (ECE) $< 0.12$.
4. **Drift Gate**: Population Stability Index (PSI) $< 0.25$.
5. **Stress Test Gate**: Performance under simulated high-volatility regimes remains within risk limits.

## 2. Model Governance Lifecycle

```
DEVELOPMENT ──► CANDIDATE ──► VALIDATING ──► VALIDATED ──► APPROVAL_REQUIRED ──► APPROVED ──► PRODUCTION
```

The active production model artifact is immutable and can never be silently overwritten by newly trained candidate models.
