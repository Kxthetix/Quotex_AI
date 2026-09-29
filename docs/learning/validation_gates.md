# The 10 Validation Gates

## 1. Overview & Purpose

Before any candidate model can even be reviewed for promotion, it must be evaluated by the `ValidationEngine` across 10 deterministic gates. Every gate produces `PASS`, `FAIL`, or `INSUFFICIENT_DATA`.

## 2. Gate Definitions

| Gate | Name | Criteria |
| --- | --- | --- |
| Gate 1 | `GATE_1_DATA_QUALITY` | Missing data rate $< 1.0\%$. |
| Gate 2 | `GATE_2_FEATURE_COMPATIBILITY` | Matches locked feature version and schema. |
| Gate 3 | `GATE_3_TRAINING_REPRODUCIBILITY` | Fixed random seed recorded and reproducible. |
| Gate 4 | `GATE_4_VALIDATION_PERFORMANCE` | Out-of-sample validation accuracy $\ge 52\%$. |
| Gate 5 | `GATE_5_TEST_PERFORMANCE` | Final out-of-sample test accuracy $\ge 50\%$. |
| Gate 6 | `GATE_6_WALK_FORWARD_STABILITY` | Rolling walk-forward mean Sharpe $\ge 0.0$ and max drawdown $< 20\%$. |
| Gate 7 | `GATE_7_CALIBRATION` | Expected Calibration Error (ECE) $\le 0.15$. |
| Gate 8 | `GATE_8_DRIFT_ROBUSTNESS` | Absence of severe feature/prediction drift. |
| Gate 9 | `GATE_9_RISK_CONSTRAINTS` | Backtested simulated max drawdown $< 20\%$. |
| Gate 10 | `GATE_10_ARTIFACT_INTEGRITY` | Artifact exists, SHA256 checksum matches, clean deserialization. |

## 3. Consolidation & Status

- If any gate fails $\rightarrow$ `ValidationResult.final_status = FAIL`.
- If no gate fails but data is missing $\rightarrow$ `INSUFFICIENT_DATA`.
- Only if all 10 gates pass $\rightarrow$ `PASS`.
