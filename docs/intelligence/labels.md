# Label Engine Specification

## 1. Overview & Separation of Concerns

The `LabelEngine` (version `v1.0.0`) calculates forward supervised targets strictly from price action occurring after observation timestamp $T$.

| Concept | Question Answered | Temporal Boundary |
| :--- | :--- | :--- |
| **Features** | "What was known at $T$?" | Data $\le T$ |
| **Labels** | "What happened after $T$?" | Data $> T$ (specifically $t+1 \dots t+H$) |

Features and labels are kept completely distinct and decoupled.

## 2. Specialized Label Definitions

The engine generates four specialized forward target classes over configurable horizons $H \in \{5, 15, 60\}$ minutes:

### 2.1 Direction Label (`direction_{H}`)
- **Target Type**: Binary classification ($0$ or $1$).
- **Definition**:
  $$\text{direction}_H = \begin{cases} 1 & \text{if } C_{t+H} > C_t \\ 0 & \text{otherwise} \end{cases}$$
- **Tail Boundary**: Rows where $t + H \ge N$ are strictly set to `NaN`.

### 2.2 Return Label (`return_{H}` / `target_return_{H}`)
- **Target Type**: Continuous regression.
- **Definition**:
  $$R_H = \frac{C_{t+H} - C_t}{C_t}$$
- **Invariant**: Forward percentage return magnitude; never included in feature vectors.

### 2.3 Volatility Label (`volatility_{H}`)
- **Target Type**: Continuous regression (non-negative).
- **Definition**: Annualized standard deviation of 1-minute forward log returns over bars $t+1 \dots t+H$:
  $$\sigma_H = \sqrt{525,600} \times \sqrt{\frac{1}{H-1} \sum_{i=1}^H (r_{t+i} - \bar{r})^2}$$

### 2.4 Trade-Quality Label (`trade_quality_{H}`)
- **Target Type**: Binary classification ($0$ or $1$).
- **Definition**: Chronologically tracks whether price hits a predefined profit target $+0.5\%$ before an invalidation stop $-0.5\%$ within horizon $H$.
  - Returns $1.0$ if high reaches entry $\times (1 + \text{target\_pct})$ before low reaches entry $\times (1 - \text{stop\_pct})$.
  - Returns $0.0$ if stop loss is hit first, or if the horizon expires without reaching the profit target.

## 3. Label Versioning Invariant

Every label is assigned an immutable version (`v1.0.0`). Any change to horizons, profit target/stop thresholds, or realized volatility formulas requires a version bump, preventing silent alterations to training targets.
