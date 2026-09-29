# Machine Learning Models & Model Registry Specification

## 1. Overview & Architecture

Phase 4 establishes specialized, single-purpose machine-learning estimators rather than a monolithic monolithic prediction engine. Each model is tailored to a specific target and governed by a local model registry.

```
                   +----------------------+
                   |    Feature Vector    |
                   +----------+-----------+
                              |
     +----------------+-------+--------+----------------+
     |                |                |                |
     v                v                v                v
+----------+    +----------+     +------------+   +---------------+
|Direction |    |  Return  |     | Volatility |   | Trade-Quality |
|Classifier|    |Regressor |     | Regressor  |   |  Classifier   |
+----------+    +----------+     +------------+   +---------------+
```

## 2. Specialized Models

### 2.1 Direction Model (`DirectionModel`)
- **Target**: `direction_{H}` (Binary: 0 for DOWN, 1 for UP).
- **Supported Algorithms**: Logistic Regression, Random Forest, LightGBM.
- **Output**: Direction class and probability distribution $[P(\text{DOWN}), P(\text{UP})]$.
- **Constraint**: Probabilities are NOT converted to trading signals (reserved for Phase 6).

### 2.2 Return Model (`ReturnModel`)
- **Target**: `target_return_{H}` (Continuous percentage return magnitude).
- **Supported Algorithms**: Ridge Regression, Random Forest, LightGBM.
- **Output**: Expected return percentage. Not a guaranteed return.

### 2.3 Volatility Model (`VolatilityModel`)
- **Target**: `volatility_{H}` (Realized annualized forward volatility).
- **Supported Algorithms**: Ridge Regression, Random Forest, LightGBM.
- **Output**: Expected annualized volatility $\hat{\sigma} \ge 0$.

### 2.4 Trade-Quality Model (`TradeQualityModel`)
- **Target**: `trade_quality_{H}` (1 if target hit before stop, else 0).
- **Supported Algorithms**: Logistic Regression, Random Forest, LightGBM.
- **Output**: Model-estimated probability of favorable trade completion.

### 2.5 Learned Regime Model (`LearnedRegimeModel`)
- **Target**: Multi-class market regime string/integer.
- **Supported Algorithms**: Random Forest, LightGBM.
- **Independence**: Kept strictly distinct and versioned independently from the rule-based `MarketRegimeEngine`.

## 3. Local Model Registry (`LocalModelRegistry`)

Persists model binaries alongside immutable metadata:
- **Location**: `data/models/`
- **Catalog**: `registry_index.json`
- **Metadata Fields**:
  - `model_id`: Unique identifier (e.g., `model_4b2c1a89f0e1`)
  - `model_name` & `model_version`
  - `feature_version` & `label_version`
  - `expected_features`: Strictly ordered list of feature column names
  - `algorithm` & `parameters`
  - `metrics`: Standardized evaluation scores
  - `training_period`: Start and end timestamps
  - `status`: Lifecycle governance state (`EXPERIMENT`, `CANDIDATE`, `VALIDATION`, `REJECTED`)
  - `created_at`: UTC timestamp

## 4. Feature Schema Compatibility Guard

Before generating predictions, every model calls:
```python
model.validate_features(feature_names, feature_version)
```
If the input feature version differs from the model's `feature_version`, or if column names or ordering deviate, the system raises `IncompatibleFeatureError` to fail safely and prevent silent inference corruption.
