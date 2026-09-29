# Phase 3 Feature Catalog

Comprehensive metadata catalog documenting all 92 deterministic, version-locked market features implemented in the Trading Intelligence System (Feature Version `v1.0.0`).

> **Design Principle**: Features answer *"What was measurable from the market data at this timestamp?"* A feature never answers *"Should I BUY or SELL?"* Zero trading signals or thresholds are embedded in feature calculations.

---

## Summary of Feature Categories

| Category | Count | Primary Inputs | Description |
|---|---|---|---|
| **Price** | 14 | OHLCV | Price changes, log returns, wicks, candle body, candle range, and price gaps |
| **Trend** | 18 | OHLCV | Exponential Moving Averages, Simple Moving Averages, ADX (+DI/-DI), linear regression slopes, structural swing states |
| **Momentum** | 11 | OHLCV | Relative Strength Index (Wilder), MACD line/signal/hist, Rate of Change (ROC), Stochastic %K/%D |
| **Volatility** | 7 | OHLCV | Average True Range (Wilder), rolling return standard deviation, annualized log return historical volatility |
| **Volume** | 8 | Volume, OHLCV | Relative volume ratio, volume moving averages, volume spike detectors, volume regression slope |
| **Structure** | 9 | High, Low, Close | Prior support and resistance levels, breakout/breakdown indicators, consolidation range compression, channel position |
| **Multi-Timeframe** | 25 | 5m, 15m, 1h, 4h, 1d | Aligned higher-timeframe EMA, RSI, ATR, and 1-bar returns strictly locked to completed bar availability times |
| **Total** | **92** | | |

---

## 1. Price Features (14 Features)

### Returns Family (`return_1`, `return_3`, `return_5`, `return_15`, `return_30`, `return_60`)
- **Category**: Price
- **Formula**: $(C_t - C_{t-N}) / C_{t-N}$
- **Parameters**: `period: N` $\in \{1, 3, 5, 15, 30, 60\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Produces `None` / `NaN` with `FeatureValidity.WARMING_UP` for $t < N$. Never silently filled with zero.
- **Output Meaning**: Simple percentage price change over the prior $N$ completed candles.
- **Known Limitations**: Sensitive to sudden outlier ticks if raw inputs are not validated.

### Log Returns Family (`log_return_1`, `log_return_5`, `log_return_15`)
- **Category**: Price
- **Formula**: $\ln(C_t / C_{t-N})$
- **Parameters**: `period: N` $\in \{1, 5, 15\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Produces `None` for $t < N$ (`WARMING_UP`).
- **Output Meaning**: Continuously compounded rate of return over $N$ bars; additive across time.
- **Known Limitations**: Requires strictly positive price inputs ($C > 0$).

### Candle Body (`candle_body`)
- **Category**: Price
- **Formula**: $C_t - O_t$
- **Parameters**: None
- **Inputs**: `close`, `open`
- **Timeframe**: Base (`1m`)
- **Lookback**: 1 bar
- **Warm-Up Behavior**: Valid immediately on the first bar.
- **Output Meaning**: Net price difference between the bar close and bar open. Positive indicates a bullish close; negative indicates a bearish close.

### Candle Wicks (`upper_wick`, `lower_wick`)
- **Category**: Price
- **Formula**:
  - `upper_wick` = $H_t - \max(O_t, C_t)$
  - `lower_wick` = $\min(O_t, C_t) - L_t$
- **Parameters**: None
- **Inputs**: `high`, `low`, `open`, `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: 1 bar
- **Warm-Up Behavior**: Valid immediately on the first bar.
- **Output Meaning**: Length of the upper and lower shadows, measuring intra-candle price rejection. Non-negative.

### Candle Range (`candle_range`)
- **Category**: Price
- **Formula**: $H_t - L_t$
- **Parameters**: None
- **Inputs**: `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 1 bar
- **Warm-Up Behavior**: Valid immediately on the first bar.
- **Output Meaning**: Total dispersion between high and low during the candle period.

### Price Gap (`gap`)
- **Category**: Price
- **Formula**: $O_t - C_{t-1}$
- **Parameters**: None
- **Inputs**: `open`, `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: 2 bars
- **Warm-Up Behavior**: Requires 1 prior bar; index 0 produces `None` (`WARMING_UP`).
- **Output Meaning**: Discontinuity between current open and preceding close.

---

## 2. Trend Features (18 Features)

### EMA Families (`ema_9`, `ema_20`, `ema_50`, `ema_100`, `ema_200`)
- **Category**: Trend
- **Formula**: $EMA_t = \alpha \cdot C_t + (1 - \alpha) \cdot EMA_{t-1}$, where $\alpha = 2 / (N + 1)$
- **Parameters**: `period: N` $\in \{9, 20, 50, 100, 200\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars (`min_periods=N`)
- **Warm-Up Behavior**: Produces `None` with `FeatureValidity.WARMING_UP` for indices $< N-1$.
- **Output Meaning**: Exponentially weighted moving average placing higher weight on recent prices.

### SMA Families (`sma_20`, `sma_50`, `sma_100`, `sma_200`)
- **Category**: Trend
- **Formula**: $\frac{1}{N} \sum_{i=0}^{N-1} C_{t-i}$
- **Parameters**: `period: N` $\in \{20, 50, 100, 200\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Produces `None` with `FeatureValidity.WARMING_UP` until $N$ full bars exist.
- **Output Meaning**: Unweighted arithmetic average of the last $N$ candle closes.

### Average Directional Index (`adx_14`, `plus_di_14`, `minus_di_14`)
- **Category**: Trend
- **Formula**: Wilder's smoothed $+DM$ and $-DM$ divided by smoothed True Range ($TR$):
  - $+DI_{14} = 100 \times (\text{Smooth}_{+DM} / \text{Smooth}_{TR})$
  - $-DI_{14} = 100 \times (\text{Smooth}_{-DM} / \text{Smooth}_{TR})$
  - $DX = 100 \times |+DI - -DI| / (+DI + -DI)$
  - $ADX_{14} = \text{Smooth}(DX, 14)$
- **Parameters**: `period: 14`
- **Inputs**: `high`, `low`, `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: 28 bars (14 for $+DI/-DI$, 28 for $ADX$)
- **Warm-Up Behavior**: Bounded in $[0, 100]$. `None` until 28 completed bars.
- **Output Meaning**: Trend strength measurement regardless of trend direction.

### Normalized Linear Regression Slopes (`slope_10`, `slope_20`)
- **Category**: Trend
- **Formula**: $\beta / \overline{Y}$, where $\beta$ is ordinary least squares slope of price vs time $x \in [0..N-1]$
- **Parameters**: `period: N` $\in \{10, 20\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Produces `None` for $t < N-1$.
- **Output Meaning**: Directional velocity of price normalized by mean price level.

### Trend Swing Structure (`higher_high_20`, `lower_low_20`, `higher_low_20`, `lower_high_20`)
- **Category**: Trend
- **Formula**: Evaluated against prior $N$ completed candles (strictly shifted by 1):
  - `higher_high` = $1.0$ if $H_t > \max(H_{t-N}..H_{t-1})$ else $0.0$
  - `lower_low` = $1.0$ if $L_t < \min(L_{t-N}..L_{t-1})$ else $0.0$
  - `higher_low` = $1.0$ if $L_t > \max(L_{t-N}..L_{t-1})$ else $0.0$
  - `lower_high` = $1.0$ if $H_t < \min(H_{t-N}..H_{t-1})$ else $0.0$
- **Parameters**: `lookback: 20`
- **Inputs**: `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars (20 prior bars + current bar)
- **Warm-Up Behavior**: Strictly `None` for $t < 20$.
- **Output Meaning**: Objective binary flags indicating structural expansion or compression relative to prior swing extremes.

---

## 3. Momentum Features (11 Features)

### Relative Strength Index (`rsi_7`, `rsi_14`, `rsi_21`)
- **Category**: Momentum
- **Formula**: $100 - (100 / (1 + RS))$, where $RS = \text{WilderEMA}(\text{gain}, N) / \text{WilderEMA}(\text{loss}, N)$
- **Parameters**: `period: N` $\in \{7, 14, 21\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Bounded in $[0, 100]$. Returns `50.0` on completely flat price series.
- **Output Meaning**: Velocity and magnitude of directional price movements.

### MACD Family (`macd_line`, `macd_signal`, `macd_hist`)
- **Category**: Momentum
- **Formula**:
  - `macd_line` = $EMA(12) - EMA(26)$
  - `macd_signal` = $EMA(\text{macd\_line}, 9)$
  - `macd_hist` = $\text{macd\_line} - \text{macd\_signal}$
- **Parameters**: `fast: 12, slow: 26, signal: 9`
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: 35 bars ($26 + 9$)
- **Warm-Up Behavior**: `None` during initial 34 bars.
- **Output Meaning**: Divergence and convergence between short- and medium-term exponential moving averages.

### Rate of Change (`roc_9`, `roc_14`, `roc_21`)
- **Category**: Momentum
- **Formula**: $100 \times (C_t - C_{t-N}) / C_{t-N}$
- **Parameters**: `period: N` $\in \{9, 14, 21\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: `None` for $t < N$.
- **Output Meaning**: Momentum percentage velocity over $N$ bars.

### Stochastic Oscillator (`stoch_k_14`, `stoch_d_14`)
- **Category**: Momentum
- **Formula**:
  - $\%K = \text{SMA}(100 \times (C_t - \min(L_{14})) / (\max(H_{14}) - \min(L_{14})), 3)$
  - $\%D = \text{SMA}(\%K, 3)$
- **Parameters**: `k_period: 14, smooth_k: 3, d_period: 3`
- **Inputs**: `high`, `low`, `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: 18 bars
- **Warm-Up Behavior**: Bounded in $[0, 100]$.
- **Output Meaning**: Location of close relative to the high-low range over the lookback window.

---

## 4. Volatility Features (7 Features)

### Average True Range (`atr_7`, `atr_14`)
- **Category**: Volatility
- **Formula**: $\text{WilderEMA}(TR, N)$, where $TR = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$
- **Parameters**: `period: N` $\in \{7, 14\}$
- **Inputs**: `high`, `low`, `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Strictly positive ($> 0$). `None` for $t < N$.
- **Output Meaning**: Absolute price volatility measuring market dispersion including overnight/inter-bar gaps.

### Rolling Return Volatility (`rolling_vol_10`, `rolling_vol_20`, `rolling_vol_30`)
- **Category**: Volatility
- **Formula**: Sample standard deviation of 1-period simple returns over window $N$
- **Parameters**: `period: N` $\in \{10, 20, 30\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Non-negative ($\ge 0$).
- **Output Meaning**: Realized dispersion of discrete bar-to-bar returns.

### Annualized Historical Volatility (`historical_vol_20`, `historical_vol_60`)
- **Category**: Volatility
- **Formula**: $\text{std}(\ln(C_t / C_{t-1})) \times \sqrt{365 \times 1440}$
- **Parameters**: `period: N` $\in \{20, 60\}$
- **Inputs**: `close`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Non-negative ($\ge 0$). `None` until $N$ bars available.
- **Output Meaning**: Annualized logarithmic return volatility for 24/7 crypto markets ($525,600$ minutes/year).

---

## 5. Volume Features (8 Features)

### Volume Moving Averages (`volume_ma_20`, `volume_ma_50`)
- **Category**: Volume
- **Formula**: $\frac{1}{N} \sum_{i=0}^{N-1} V_{t-i}$
- **Parameters**: `period: N` $\in \{20, 50\}$
- **Inputs**: `volume`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Non-negative.

### Relative Volume (`relative_vol_10`, `relative_vol_20`, `relative_vol_50`)
- **Category**: Volume
- **Formula**: $V_t / \overline{V}_{t-N..t-1}$ (strictly shifted by 1 to exclude current volume from baseline)
- **Parameters**: `period: N` $\in \{10, 20, 50\}$
- **Inputs**: `volume`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N + 1$ bars
- **Warm-Up Behavior**: Handles zero baseline safely by outputting `None` rather than `inf`.
- **Output Meaning**: Current volume as a ratio of prior average volume. Ratio $> 1.0$ indicates above-average volume.

### Volume Spike (`volume_spike`)
- **Category**: Volume
- **Formula**: $1.0$ if $V_t > 2.0 \times \overline{V}_{t-20..t-1}$ else $0.0$
- **Parameters**: `multiplier: 2.0, lookback: 20`
- **Inputs**: `volume`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars
- **Warm-Up Behavior**: Strictly `None` for $t < 20$.
- **Output Meaning**: Binary flag identifying extraordinary volume relative to the rolling 20-period baseline.

### Volume Trends (`volume_trend_10`, `volume_trend_20`)
- **Category**: Volume
- **Formula**: Normalized linear regression slope of volume over lookback $N$
- **Parameters**: `period: N` $\in \{10, 20\}$
- **Inputs**: `volume`
- **Timeframe**: Base (`1m`)
- **Lookback**: $N$ bars
- **Warm-Up Behavior**: Produces `None` during warm-up.
- **Output Meaning**: Rate of volume expansion or contraction over time.

---

## 6. Market Structure Features (9 Features)

### Support & Resistance Levels (`support_level_20`, `resistance_level_20`)
- **Category**: Structure
- **Formula**:
  - `support` = $\min(L_{t-20}..L_{t-1})$
  - `resistance` = $\max(H_{t-20}..H_{t-1})$
- **Parameters**: `lookback: 20`
- **Inputs**: `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars (strictly shifted by 1 to prevent current bar leakage)
- **Warm-Up Behavior**: `None` for $t < 20$.
- **Output Meaning**: Concrete price levels marking the highest high and lowest low of the prior 20 completed candles.

### Breakout & Breakdown States (`breakout_20`, `breakdown_20`)
- **Category**: Structure
- **Formula**:
  - `breakout` = $1.0$ if $C_t > \text{resistance\_level\_20}$ else $0.0$
  - `breakdown` = $1.0$ if $C_t < \text{support\_level\_20}$ else $0.0$
- **Parameters**: `lookback: 20`
- **Inputs**: `close`, `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars
- **Warm-Up Behavior**: Evaluates to `None` when support or resistance is in warm-up.
- **Output Meaning**: Binary state indicating whether the bar closed outside the previously established channel.

### Consolidation & Range Compression (`range_compression_20`, `is_consolidating_20`)
- **Category**: Structure
- **Formula**:
  - `range_compression` = $(\text{resistance} - \text{support}) / \text{mean\_price}$
  - `is_consolidating` = $1.0$ if $\text{range\_compression} < 0.02$ else $0.0$
- **Parameters**: `lookback: 20, threshold: 0.02`
- **Inputs**: `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars
- **Warm-Up Behavior**: Evaluates to `None` during initial 20 bars.
- **Output Meaning**: Objective, quantitative measurement of volatility contraction and channel tightness.

### Range Position (`range_position_20`)
- **Category**: Structure
- **Formula**: $\text{clip}((C_t - \text{support}) / (\text{resistance} - \text{support}), 0.0, 1.0)$
- **Parameters**: `lookback: 20`
- **Inputs**: `close`, `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars
- **Warm-Up Behavior**: Bounded in $[0.0, 1.0]$.
- **Output Meaning**: Relative location of the current close within the active support-resistance channel ($0.0$ = at support, $1.0$ = at resistance).

### Retest Distance Metrics (`retest_dist_resistance_20`, `retest_dist_support_20`)
- **Category**: Structure
- **Formula**:
  - $|C_t - \text{resistance}| / \text{resistance}$
  - $|C_t - \text{support}| / \text{support}$
- **Parameters**: `lookback: 20`
- **Inputs**: `close`, `high`, `low`
- **Timeframe**: Base (`1m`)
- **Lookback**: 21 bars
- **Warm-Up Behavior**: Non-negative. `None` during warm-up.
- **Output Meaning**: Normalized percentage distance between current close and structural boundaries for evaluating potential retest scenarios.

---

## 7. Multi-Timeframe Features (25 Features)

Computed across 5 configurable higher timeframes: `5m`, `15m`, `1h`, `4h`, `1d`.

Each timeframe exposes 5 version-locked features:
1. `mtf_{tf}_ema_20`: Higher-timeframe 20-period Exponential Moving Average
2. `mtf_{tf}_ema_50`: Higher-timeframe 50-period Exponential Moving Average
3. `mtf_{tf}_rsi_14`: Higher-timeframe 14-period Relative Strength Index
4. `mtf_{tf}_atr_14`: Higher-timeframe 14-period Average True Range
5. `mtf_{tf}_return_1`: Higher-timeframe 1-bar simple return

### Anti-Leakage Alignment Rules
- **Completed Bar Selection**: A higher-timeframe bar is only eligible for feature aggregation when its theoretical completion time $\le T$ (where $T$ is the base prediction timestamp).
- **Exact Boundary**: At $T = 11:00:00$, the 1-hour candle spanning $10:00:00 \to 11:00:00$ completed at $11:00:00$, and is therefore available.
- **Before Boundary**: At $T = 10:59:00$, the $10:00 \to 11:00$ candle is still in progress. The Feature Engine backward-merges the prior completed 1-hour candle ($09:00 \to 10:00$).
- **Incomplete Candles**: In-progress candles at the end of the input stream are strictly discarded during resampling and never incorporated into indicator calculations.
