# Baseline Statistical Models Specification

## 1. Overview & Purpose

Before training machine-learning models, the system evaluates transparent statistical baselines. The objective is to verify whether complex machine learning outperforms parsimonious empirical baselines on historical financial series.

## 2. Implemented Statistical Baselines

### 2.1 MajorityClassModel
- **Domain**: Binary classification (`direction`, `trade_quality`).
- **Mechanism**: Calculates empirical class frequencies $P(y=0)$ and $P(y=1)$ on training data.
- **Inference**:
  - `predict()`: Returns the majority class constant.
  - `predict_proba()`: Returns the historical training class distribution $[P(0), P(1)]$.

### 2.2 HistoricalMeanModel
- **Domain**: Continuous regression (`return`, `volatility`).
- **Mechanism**: Computes the sample mean $\mu_{train}$ and median $m_{train}$ over the training period.
- **Inference**:
  - `predict()`: Outputs the historical training mean $\mu_{train}$.

### 2.3 RegimeConditionalModel
- **Domain**: Regime-aware regression benchmark.
- **Mechanism**: Partitions training targets by the active market regime ($regime \in \text{MarketRegime}$) and computes conditional empirical expectations:
  $$\hat{y}(r) = \mathbb{E}[y \mid \text{Regime} = r]$$
- **Inference**:
  - `predict(X, regimes)`: Outputs the historical expectation conditioned on the active regime at time $T$.

## 3. BaselineModel Common Interface

All statistical baselines implement the standard `BaselineModel` abstract base class:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `validate_features(feature_names, feature_version)`
- `metadata()`
- `save(directory)`
- `load(path)`
