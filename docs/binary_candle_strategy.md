# Candle Trend-Pullback Strategy & Payout-Aware Evaluation (Research Platform)

> [!NOTE]
> This document details the research and paper-trading architecture of the binary candle strategy in `AI-Trader`. The system does **not** connect to live broker execution interfaces, does **not** risk real capital, and **never promises a high win rate or profit**. Binary options involve inherently unfavorable statistical expectancy due to asymmetric payout structures.

---

## 1. System Scope & Core Principles

The strategy is designed as an explicit, rule-based algorithmic framework to evaluate short-horizon price action using completed OHLCV bars.

1. **Deterministic & Auditable Rules**: Every entry and rejection decision is traceable through programmatic flags and diagnostic strings.
2. **Strict Non-Lookahead Execution**: Decisions are formed solely on closed candles. Simulated entries take place on the subsequent bar open, and settlements match the exact contractual expiry.
3. **Payout-Aware Mathematical Discipline**: Win rates are evaluated strictly against payout-adjusted break-even thresholds.
4. **Capped Fixed Sizing**: Martingale scaling, loss-recovery sizing, and progressive stakes are strictly prohibited.
5. **Separation of Synthetic and Historical Data**: Synthetic candles are restricted to unit test verification and smoke tests.

---

## 2. Candle Trend-Pullback Strategy Rules

The strategy evaluates whether a directional trend is pausing into a value zone (the 21-period EMA) and confirming momentum resumption.

### A. Permitted Entry Conditions

For an entry signal (`CALL` or `PUT`) to trigger, **all** of the following conditions must simultaneously align:

| Component | `CALL` (Long / Up) Requirement | `PUT` (Short / Down) Requirement |
|---|---|---|
| **Trend Hierarchy** | EMA-9 > EMA-21 and (EMA-21 > EMA-50 or EMA-50 is None). Trend state = `BULLISH`. | EMA-9 < EMA-21 and (EMA-21 < EMA-50 or EMA-50 is None). Trend state = `BEARISH`. |
| **EMA Separation** | Relative difference `abs(EMA-9 - EMA-21) / Price >= threshold` (default `0.0001`). | Relative difference `abs(EMA-9 - EMA-21) / Price >= threshold` (default `0.0001`). |
| **Pullback Retest** | At least one candle in prior 4 bars `[-5:-1]` has its low touch between `EMA-21 - 0.75 ATR` and `EMA-21 + 0.25 ATR`. | At least one candle in prior 4 bars `[-5:-1]` has its high touch between `EMA-21 - 0.25 ATR` and `EMA-21 + 0.75 ATR`. |
| **Candle Confirmation** | Latest candle is green (`close > open`), closes higher than prior candle's high (`close > high_{t-1}`), and closes in upper 35% of its range (`close_location >= 0.65`). | Latest candle is red (`close < open`), closes lower than prior candle's low (`close < low_{t-1}`), and closes in lower 35% of its range (`close_location <= 0.35`). |
| **Momentum Direction** | RSI-14 between 50.0 and `rsi_overbought` (default `68.0`). MACD histogram > 0 and `MACD_hist_{t} >= MACD_hist_{t-1}`. | RSI-14 between `rsi_oversold` (default `32.0`) and 50.0. MACD histogram < 0 and `MACD_hist_{t} <= MACD_hist_{t-1}`. |
| **Mean Extension Guard** | Distance `abs(Price - EMA-21) <= 1.5 ATR`. Price must not be overextended from the mean. | Distance `abs(Price - EMA-21) <= 1.5 ATR`. Price must not be overextended from the mean. |
| **Volatility Quality** | ATR-14 must be strictly finite and positive (`ATR > 0`). | ATR-14 must be strictly finite and positive (`ATR > 0`). |
| **Envelope Bounds** | Price must be below Upper Bollinger Band (`BB_UPPER_PIERCED` not active). | Price must be above Lower Bollinger Band (`BB_LOWER_PIERCED` not active). |
| **Candle Structure** | Candle must not be a Doji (`body_pct >= 0.06`). | Candle must not be a Doji (`body_pct >= 0.06`). |
| **Confidence Threshold** | Net composite score >= `min_confidence` (default `0.65`). | Net composite score <= `-min_confidence` (default `-0.65`). |

---

### B. Mandatory `NO_TRADE` Rejection Catalog

When any individual rule condition fails, the engine returns `BinaryAction.NO_TRADE` and attaches explicit flags to `rejection_flags`:

| Rejection Flag | Trigger Reason | Risk Managed |
|---|---|---|
| `INCOMPLETE_CANDLE` | Input series contains an unclosed or currently forming candle. | Lookahead bias & mid-candle volatility repaint. |
| `INSUFFICIENT_DATA` | Fewer than 25 historical bars available for indicator stabilization. | Cold-start calculation distortion. |
| `INVALID_VOLATILITY` | ATR-14 is zero or non-finite (flat market or zero-tick range). | Illiquid or non-trading market sessions. |
| `TREND_CONSOLIDATION` | EMAs are flat, intertwined, or lack directional hierarchy. | Choppy range-bound whipsaws. |
| `NO_PULLBACK_SETUP` | No bar in the 4-candle lookback retested the 21-period EMA band. | Chasing price at extreme breakout points. |
| `NO_CANDLE_CONFIRMATION` | Closed candle failed to exceed prior candle's high/low or lacked decisive close location. | False breakouts and weak momentum. |
| `MOMENTUM_MISALIGNED` | RSI or MACD histogram slope does not confirm the directional bias. | Counter-trend divergence trap. |
| `PRICE_EXTENDED_FROM_MEAN` | Price is more than 1.5 ATR away from EMA-21. | Exhaustion and mean-reversion snapbacks. |
| `RSI_OVERBOUGHT` | RSI-14 >= overbought limit (`68.0`). Vetoes `CALL`. | Buying the top of a move. |
| `RSI_OVERSOLD` | RSI-14 <= oversold limit (`32.0`). Vetoes `PUT`. | Selling the bottom of a drop. |
| `BB_UPPER_PIERCED` | Price closed above upper Bollinger Band (+2.0 std). | Volatility stretch exhaustion. |
| `BB_LOWER_PIERCED` | Price closed below lower Bollinger Band (-2.0 std). | Volatility stretch exhaustion. |
| `DOJI_INDECISION` | Candle body represents < 6% of total range. | Equilibrium / indecisive market balance. |
| `CROWD_TRAP_BULLISH` | Retail sentiment is extreme (>85% bullish) into resistance. | Contrarian retail trap protection. |
| `CROWD_TRAP_BEARISH` | Retail sentiment is extreme (<15% bullish) into support. | Contrarian retail trap protection. |
| `LOW_CONFIDENCE` | Composite score magnitude is below `min_confidence` (e.g. `< 0.65`). | Marginal or low-conviction setups. |

---

## 3. Strict Execution Timing & Non-Lookahead Simulation

To prevent simulation artifacts and lookahead bias, the backtesting engine (`backtesting/binary_options.py`) enforces strict timing rules:

```
 Candle t (Signal Bar)          Candle t+1 (Entry Bar)          Candle t+K (Settlement Bar)
┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
│ Open   ...   Close    │      │ Open   ...   Close    │      │ Open   ...   Close    │
└───────────────────────┘      └───────────────────────┘      └───────────────────────┘
▲                       ▲      ▲                                                      ▲
│                       │      │                                                      │
start_time           end_time  start_time                                         end_time
                        │      │                                                      │
              Signal Evaluated └────── Trade Entered                          Trade Settled
              (Only on Closed)         (At Open Price)                        (At Close Price)
```

1. **Signal Availability**: A signal is evaluated *only* after candle $t$ has fully closed at `signal_bar.end_time`.
2. **Simulated Entry**: The trade enters at `entry_bar.open` where `entry_bar.start_time >= signal_bar.end_time`. Look-ahead timing is strictly checked: if `entry_bar.start_time < signal_bar.end_time`, a `ValueError` is raised.
3. **Settlement**: Settlement occurs at `settlement_bar.close` on bar $t + expiry\_bars$ at `settlement_bar.end_time`.
4. **Single Position Constraint**: Only one trade is open at a time. The next signal evaluation occurs only on or after the bar where the prior trade settled.

---

## 4. Payout-Aware Binary Mathematics & Risk Management

### A. Break-Even Win Rate Formula

In fixed-expiry binary options, a winning trade returns the stake plus a net payout fraction $r$, while a losing trade forfeits the entire stake. The net expected value per trade is:

$$E = W \cdot (r \cdot S) - (1 - W) \cdot S$$

Where:
- $W$ = Win rate (probability of winning)
- $r$ = Net payout rate (e.g., $0.85$ for $85\%$ payout, $0.93$ for $93\%$ payout)
- $S$ = Fixed stake

Setting $E = 0$ yields the **Break-Even Win Rate**:

$$W_{BE} = \frac{1}{1 + r}$$

#### Reference Table of Break-Even Rates

| Broker Payout Rate ($r$) | Break-Even Win Rate ($W_{BE}$) | Minimum Win Rate to Beat Friction ($+2\%$) |
|---|---|---|
| **70%** (`0.70`) | **58.82%** | 60.82% |
| **75%** (`0.75`) | **57.14%** | 59.14% |
| **80%** (`0.80`) | **55.56%** | 57.56% |
| **85%** (`0.85`) | **54.05%** | 56.05% |
| **90%** (`0.90`) | **52.63%** | 54.63% |
| **92%** (`0.92`) | **52.08%** | 54.08% |
| **93%** (`0.93`) | **51.81%** | 53.81% |
| **95%** (`0.95`) | **51.28%** | 53.28% |

> [!IMPORTANT]
> If a strategy achieves a $52\%$ win rate on an asset with an $85\%$ payout ($W_{BE} = 54.05\%$), the expectancy is **negative**, resulting in systematic capital loss despite winning more than half the trades.

### B. Tie Handling
Equal settlement price (`settlement_price == entry_price`) is conservatively treated as a **loss** unless the specific broker contract explicitly specifies an exact-tie refund.

### C. Fixed and Capped Stake Sizing (Anti-Martingale)
- **Fixed Sizing**: Every trade risks an identical, constant stake $S$.
- **Exposure Cap**: $S \le \text{starting\_balance} \times \text{max\_stake\_fraction}$ (default $\le 1\%$).
- **No Martingale**: Increasing stake size after losses or doubling down is strictly prohibited. Martingale leads to exponential capital depletion and catastrophic drawdown.

---

## 5. Research Methodology & Data Limitations

### A. Data Matching Requirement
A strategy designed for 1-minute binary options **must** be evaluated against:
1. The **exact same instrument** (e.g., `GBP/JPY` or OTC pair).
2. The **exact same candle timeframe** (e.g., `1m`).
3. The **exact contract expiry** (e.g., 1 bar).
4. The **exact broker payout rate** (e.g., $0.92$).

### B. Fundamental Limitations of OTC Binary Markets
- **No Public Audited History**: OTC binary brokers operate internal, dealer-quoted synthetic price feeds that differ from interbank spot feeds. Public historical tick archives for these feeds do not exist.
- **Asymmetric Payouts & Drift**: Payout rates are dynamic and often decline during high-probability setups or volatile news periods.
- **Proxy Invalidation**: Testing on spot exchange data (e.g., Binance `BTCUSDT` 1-hour candles) serves as a software-pipeline check only. It cannot and must not be used to claim that a 1-minute OTC binary scalping strategy is profitable.

### C. Quarantine of Synthetic Smoke-Test Data
Synthetic candles generated by `generate_synthetic_candles` are strictly for pipeline verification. They are automatically flagged with `[SMOKE-TEST ONLY]` banners and must never be stored in performance audit reports.

---

## 6. CLI Usage

### Chronological Out-of-Sample Backtest
```powershell
python scripts/backtest_binary_options.py --data-file data/datasets/market_regimes/covid_crash_2020_BTCUSDT_1h_ohlcv.parquet --payout 0.85 --expiry-bars 1 --oos-fraction 0.30
```

### Rule & Indicator Inspector (Smoke Test)
```powershell
python scripts/run_binary_signal_evaluator.py --asset EUR/CHF
```
