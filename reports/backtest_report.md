# Historical Backtest Report: `bt_581958b351`

**Execution Timestamp:** 2026-09-28 20:07:37 UTC
**Code Version:** `v1.0.0` | **Model Version:** `v1.0.0` | **Feature Version:** `v1.0.0`

---

## 1. Executive Summary

| Metric | Result | Benchmark (Buy & Hold) |
| :--- | :--- | :--- |
| **Initial Capital** | $10,000.00 | $10,000.00 |
| **Final Equity** | $9,869.88 | - |
| **Total Return** | **-1.30%** | **+0.69%** |
| **CAGR** | -1.30% | - |
| **Max Drawdown** | **1.32%** | - |
| **Sharpe Ratio** | -111.39 | - |
| **Sortino Ratio** | -67.04 | - |
| **Calmar Ratio** | -0.98 | - |
| **Win Rate** | 0.0% | - |
| **Profit Factor** | 0.00 | - |
| **Expectancy** | $-16.27 | - |

---

## 2. Test Configuration & Execution Assumptions

- **Symbol:** `BTC/USDT`
- **Timeframe:** `1m`
- **Period:** 2026-01-01 00:01 to 2026-01-01 08:20 (8:19:00)
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
| **Gross Realized P&L** | $-50.67 | _Pre-friction return_ |
| **Exchange Fees** | $79.46 | -156.8% |
| **Spread Friction** | $15.89 | -31.4% |
| **Execution Slippage** | $47.67 | -94.1% |
| **Net Realized P&L** | **$-130.12** | **Net realizable performance** |

---

## 4. Trade Statistics

- **Total Completed Trades:** 8
- **Winning Trades:** 0 (0.0%)
- **Losing Trades:** 8
- **Average Trade Net P&L:** $-16.27
- **Average Winning Trade:** $0.00
- **Average Losing Trade:** $16.27
- **Average Holding Period:** 32.1 minutes

---

## 5. Regime Performance Breakdown

| Regime | Trades | Net P&L ($) | Win Rate (%) | Avg Trade ($) |
| :--- | :--- | :--- | :--- | :--- |
| SIDEWAYS | 8 | $-130.12 | 0.0% | $-16.27 |

---

## 6. Data Quality & Diagnostic Warnings

- _No data-quality anomalies detected._
