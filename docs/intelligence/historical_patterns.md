# Historical Pattern Engine Specification

## 1. Overview & Objective

The `HistoricalPatternEngine` (version `v1.0.0`) performs non-parametric historical analog discovery. It searches past market states to answer:
*"When the market previously exhibited similar feature conditions, what was the empirical distribution of subsequent price outcomes?"*

## 2. Methodology & State Representation

1. **State Vector**: At query time $T$, the engine constructs a state vector using canonical Phase 3 features (RSI, ADX, DI+, DI-, EMA distance, volume, volatility).
2. **Standardized Distance Search**: Features are standardized using historical population standard deviations. Similarity is computed via normalized Euclidean distance:
   $$D(x, y) = \sqrt{\frac{1}{K} \sum_{k=1}^K \left( \frac{x_k - y_k}{\sigma_k + \epsilon} \right)^2}$$
3. **Similarity Score**:
   $$\text{Similarity}(x, y) = \frac{1.0}{1.0 + D(x, y)} \in (0.0, 1.0]$$

## 3. Sequence-Pattern Matching

The engine supports multi-candle sequence matching across configurable windows:
- 20 candles
- 50 candles
- 100 candles

Sequences are compared using rolling normalized Euclidean distance over the trajectory of normalized returns, ensuring structural path similarity is captured alongside endpoint features.

## 4. Anti-Leakage Guard (Section 14)

When querying at time $T$ with forward horizon $H$:
A historical observation at $t_{match}$ is considered an eligible historical candidate **if and only if**:
$$t_{match} + H \le T$$

This invariant guarantees that:
1. Similarity search uses ONLY information known by $t_{match}$.
2. The subsequent outcome (forward return, MFE, MAE) was fully completed before $T$, ensuring zero leakage of future market data into the historical reference set.

## 5. Conditional Outcome Distributions

For the top $K$ nearest historical matches, the engine computes empirical conditional statistics:
- Match count & mean similarity score
- Forward return mean, median, standard deviation
- Maximum Favorable Excursion (MFE) mean
- Maximum Adverse Excursion (MAE) mean
- Positive outcome frequency (win rate)
- Percentiles (p10, p25, p50, p75, p90)

These distributions represent empirical historical evidence rather than deterministic trading guarantees.
