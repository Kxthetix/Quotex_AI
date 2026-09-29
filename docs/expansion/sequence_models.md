# Advanced Sequence Models

## 1. Sequence Dataset Construction

The `SequenceDatasetBuilder` transforms tabular feature matrices into chronological 3D sliding windows:
$$(N, \text{sequence\_length}, \text{num\_features})$$
- Target $y_i$ is aligned with the outcome following observation $T$.
- Window $i$ strictly contains rows from $t_{\text{start}}$ to $t_{\text{end}} = T$.

## 2. Model Architecture: `TemporalSequenceModel`

Conforming to the canonical `BaselineModel` interface:
1. **Exponential Recency Weighting**: Weights older sequence steps with decay parameter $\alpha = 0.95$, emphasizing recent price velocity.
2. **Velocity Feature Extraction**: Measures $\Delta = X_T - X_{T - L + 1}$ across sequence steps.
3. **Probability Calibration**: Generates well-calibrated directional probabilities for the Signal Engine.

## 3. Multi-Model Ensemble Layer

The `EnsembleModel` unifies multiple sub-models:
$$P(\text{BUY} \mid X) = \sum_{k=1}^M w_k \cdot P_k(\text{BUY} \mid X), \quad \sum w_k = 1.0$$
Component model versions and weights are preserved in metadata for governance auditing.
