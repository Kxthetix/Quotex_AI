# Drift Detection Framework

## 1. Overview & Separation of Drift

Phase 8 strictly differentiates between distinct categories of drift:
- **Feature Drift (Data Drift)**: Shifts in input distributions $P(X)$.
- **Prediction Drift**: Shifts in model output distributions $P(\hat{Y})$.
- **Concept Drift**: Changes in the relationship between inputs and ground-truth outcomes $P(Y \mid X)$.
- **Regime Drift**: Macro market environment shifts.

These dimensions are tracked separately and are **never** collapsed into a single ambiguous drift score.

## 2. Feature Drift Metrics

- **Population Stability Index (PSI)**:
  $$\text{PSI} = \sum (P_i - Q_i) \times \ln(P_i / Q_i)$$
  - $\text{PSI} < 0.10$: `NO_DRIFT`
  - $0.10 \le \text{PSI} < 0.25$: `MODERATE_DRIFT`
  - $\text{PSI} \ge 0.25$: `SEVERE_DRIFT`
- **Missing Value Shifts**: Tracks missing value rates between baseline and live data.

## 3. Prediction Drift Metrics

- Output confidence PSI
- Forward expected return distribution PSI
- Signal action frequency proportions (`BUY` / `SELL` / `NO_TRADE`)

## 4. Concept Drift Metrics

- Directional accuracy drop: $\text{Drop} = \text{Accuracy}_{\text{ref}} - \text{Accuracy}_{\text{cur}}$
- Prediction error increase ratio: $\text{Ratio} = \text{MAE}_{\text{cur}} / \text{MAE}_{\text{ref}}$
- Realized return decay across evaluated predictions
