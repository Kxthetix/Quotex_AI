# Model Auditing Methodology

## 1. Overview & Objectives

The `ModelAuditor` conducts independent, non-destructive evaluations of active production models over configurable observation windows (e.g. rolling 7 days, 30 days, or rolling N predictions).

## 2. Minimum Sample Size Requirements (Section 10)

To prevent false conclusions from small sample noise, the auditor enforces strict minimum observations:
- Default `min_audit_sample_size = 30` evaluated predictions.
- If sample size < threshold, status is `INSUFFICIENT_DATA`.
- The model is **NEVER** marked as degraded or failed simply due to insufficient data.

## 3. Auditing Dimensions

1. **Directional Accuracy**: Compares binary predicted movement against realized price returns.
2. **Prediction Error**: Mean Absolute Error (MAE) and Mean Forecast Error (MFE) across forward horizons.
3. **Probability Calibration**: Evaluates Expected Calibration Error (ECE) and Brier score.
4. **Regime-Specific Performance**: Breaks down metrics across `TREND_UP`, `TREND_DOWN`, `SIDEWAYS`, `HIGH_VOLATILITY`, and `LOW_VOLATILITY`.
5. **Drift Metrics**: Consolidates feature, prediction, concept, and regime drift indicators.

## 4. Recommendation States

- `HEALTHY`: Nominal performance; metrics exceed operational thresholds.
- `MONITOR`: Mild drift or transient dip observed; continue observation.
- `REVIEW`: Sustained degradation or calibration decay; requires human inspection.
- `RETRAIN_CANDIDATE`: Severe degradation or concept drift justifies retraining a candidate model.
- `INSUFFICIENT_DATA`: Observation count below required threshold.
