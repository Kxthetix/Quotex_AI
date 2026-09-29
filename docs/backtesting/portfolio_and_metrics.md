# Phase 5 — Portfolio Accounting and Performance Metrics

## 1. Position and Portfolio Engine

### Position Engine (`backtesting/position.py`)
- Tracks active position quantity, direction (`LONG`, `SHORT`, `FLAT`), and volume-weighted average entry price.
- Supports position opening, scaling in (increasing), scaling out (partial reductions), and full exits.
- Produces immutable `TradeRecord` instances upon position closure.

### Portfolio Engine (`backtesting/portfolio.py`)
- Strict asset-liability cash accounting:
  $$\text{Equity} = \text{Cash} + \sum \text{Market Value of Positions}$$
- Maintains running peak equity and instantaneous drawdown percentage:
  $$\text{Drawdown} = \frac{\text{Peak Equity} - \text{Current Equity}}{\text{Peak Equity}}$$
- Snapshots chronological `EquityPoint` records for downstream equity curve export and visualization.

## 2. Comprehensive Performance Metrics (`backtesting/metrics.py`)

1. **Total Return:** $\frac{\text{Final Equity} - \text{Initial Capital}}{\text{Initial Capital}}$
2. **CAGR:** Compound Annual Growth Rate over the test horizon.
3. **Maximum Drawdown & Duration:** Peak-to-trough decline across the continuous equity curve and peak recovery duration.
4. **Annualized Sharpe Ratio:** Risk-adjusted excess return over standard deviation:
   $$\text{Sharpe} = \frac{\mu_{excess}}{\sigma} \times \sqrt{N_{periods}}$$
5. **Annualized Sortino Ratio:** Downside-deviation-adjusted return:
   $$\text{Sortino} = \frac{\mu_{excess}}{\sigma_{downside}} \times \sqrt{N_{periods}}$$
6. **Calmar Ratio:** $\frac{\text{CAGR}}{|\text{Max Drawdown}|}$ with safe zero-drawdown handling.
7. **Win Rate:** Percentage of completed trades with positive net P&L.
8. **Profit Factor:** $\frac{\text{Gross Winning Profit}}{|\text{Gross Losing Loss}|}$ with safe zero-loss handling.
9. **Expectancy:**
   $$\text{Expectancy} = (\text{Win Rate} \times \text{Avg Win}) - ((1 - \text{Win Rate}) \times \text{Avg Loss})$$
10. **Benchmark Comparison:** Compares active strategy total return against Buy-and-Hold benchmark over identical candle horizons.
11. **Cost Breakdown:** Separately reports Gross P&L, Fees, Spread Cost, Slippage Cost, and Net Realized P&L.
12. **Regime Analysis:** Breaks down trade count, return, win rate, and average trade by market regime (`TREND_UP`, `TREND_DOWN`, `SIDEWAYS`, etc.).
