# Historical Backtest Report: `bt_3b916803ad`

**Execution Timestamp:** 2026-10-06 20:38:45 UTC
**Code Version:** `v1.0.0` | **Model Version:** `v1.0.0` | **Feature Version:** `v1.0.0`

---

## 1. Executive Summary

| Metric | Result | Benchmark (Buy & Hold) |
| :--- | :--- | :--- |
| **Initial Capital** | $10,000.00 | $10,000.00 |
| **Final Equity** | $10,440.57 | - |
| **Total Return** | **+4.41%** | **-24.45%** |
| **CAGR** | +69.03% | - |
| **Max Drawdown** | **14.61%** | - |
| **Sharpe Ratio** | 8.03 | - |
| **Sortino Ratio** | 6.50 | - |
| **Calmar Ratio** | 4.73 | - |
| **Win Rate** | 31.8% | - |
| **Profit Factor** | 1.23 | - |
| **Expectancy** | $20.03 | - |

---

## 2. Test Configuration & Execution Assumptions

- **Symbol:** `BTCUSDT`
- **Timeframe:** `1h`
- **Period:** 2020-03-01 00:59 to 2020-03-31 00:59 (30 days, 0:00:00)
- **Fee Model:** Maker: `2.0 bps` | Taker: `5.0 bps` | Fixed: `$0.00`
- **Spread Model:** Half-Spread `1.0 bps` (Bid/Ask derived around mark)
- **Slippage Model:** Baseline `3.0 bps`
- **Latency Model:** `0.0s` simulation execution delay
- **Liquidity Model:** Max bar volume capacity `10.0%` (Partial fills: `True`)
- **Position Sizing:** `PERCENT_CAPITAL` (Value: `0.1`, Max Size: `5.0`, Max Notional: `$50,000.00`)
- **End-of-Test Action:** `FORCE_CLOSE`

---

## 3. Cost & Friction Breakdown

| Cost Category | Total Amount ($) | Impact on P&L |
| :--- | :--- | :--- |
| **Gross Realized P&L** | $+600.09 | _Pre-friction return_ |
| **Exchange Fees** | $159.52 | -26.6% |
| **Spread Friction** | $31.90 | -5.3% |
| **Execution Slippage** | $95.71 | -15.9% |
| **Net Realized P&L** | **$+440.57** | **Net realizable performance** |

---

## 4. Trade Statistics

- **Total Completed Trades:** 22
- **Winning Trades:** 7 (31.8%)
- **Losing Trades:** 15
- **Average Trade Net P&L:** $20.03
- **Average Winning Trade:** $335.81
- **Average Losing Trade:** $127.34
- **Average Holding Period:** 954.5 minutes

---

## 5. Regime Performance Breakdown

| Regime | Trades | Net P&L ($) | Win Rate (%) | Avg Trade ($) |
| :--- | :--- | :--- | :--- | :--- |
| SIDEWAYS | 22 | $440.57 | 31.8% | $20.03 |

---

## 6. Data Quality & Diagnostic Warnings

- _No data-quality anomalies detected._
