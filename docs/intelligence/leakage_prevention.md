# Data Leakage Prevention & Reproducibility Controls

## 1. Mandatory Leakage Controls

Preventing lookahead bias and data leakage is paramount in quantitative trading systems. Phase 4 enforces strict temporal isolation across the entire pipeline:

```
[ PAST CANDLES (<= T) ] -----------> [ FEATURES (T) ]
                                             |
                                             v
                                     [ MODEL INPUT (T) ]
                                             |
[ FUTURE CANDLES (> T) ] -----------> [ TARGET LABEL (T+H) ]
```

### 1.1 Critical Invariants
1. **Features at $T$**: Depend exclusively on market data occurring $\le T$.
2. **Labels at $T$**: Depend exclusively on market data occurring after $T$ ($t+1 \dots t+H$). Future candles cannot alter past features.
3. **Historical Pattern Matching at $T$**: A past match at $t_{match}$ is valid only if $t_{match} + H \le T$. Future outcomes of historical matches must have been fully realized prior to query time $T$.
4. **Chronological Splitting**: No random or shuffled train/test splitting. Partitions enforce:
   $$\max(T_{\text{train}}) < \min(T_{\text{val}}) < \min(T_{\text{test}})$$
5. **Preprocessing Isolation (`ChronologicalPreprocessor`)**:
   - `StandardScaler` mean and variance are learned **only** on the training split.
   - Missing-value medians are calculated **only** on the training split.
   - Validation and test splits are transformed using frozen training parameters.
6. **Calibration Isolation**: Probability calibrators are fitted strictly on validation split predictions, never on training or test sets.

## 2. Automated Anti-Leakage Verification Tests

The test suite includes dedicated automated leakage verification:

- **Section 47 Critical Label Test (`tests/unit/intelligence/test_leakage_critical.py`)**:
  - Generates features and labels at time $T$.
  - Appends future candles to the series.
  - Recalculates features across the extended series.
  - Formally asserts that feature values at time $T$ are bit-for-bit identical before and after future data was introduced.
- **Section 48 Critical Train/Test Test**:
  - Validates sequential partition boundaries without temporal overlap.
  - Perturbs test features with extreme outliers and verifies that preprocessor internal states (means, scales, medians) remain strictly unchanged.

## 3. Reproducibility Guarantee

All experiments are reproducible under the invariant tuple:
$$\langle \text{Dataset ID}, \text{Feature Version}, \text{Label Version}, \text{Configuration}, \text{Model Parameters}, \text{Random Seed} \rangle$$
Setting fixed random seeds across estimators (e.g., `random_state=42`) guarantees deterministic results across multiple training runs.
