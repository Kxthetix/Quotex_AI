# Rolling Walk-Forward Validation Report

**Windows Evaluated:** 3
**Leakage Invariant Status:** **PASSED (Strict Monotonic Separation)**
**Window Sizing:** Train=400 bars, Val=100 bars, Test=100 bars, Step=100 bars

---

## 1. Aggregate Out-Of-Sample Performance

| Metric | Aggregate Out-of-Sample Result |
| :--- | :--- |
| **Combined OOS Trades** | 5 |
| **Combined OOS Return** | **-0.15%** |
| **Out-of-Sample Sharpe Ratio** | **-21.17** |
| **Out-of-Sample Sortino Ratio** | -12.63 |
| **Out-of-Sample Max Drawdown** | 0.55% |
| **Out-of-Sample Win Rate** | 40.0% |
| **Out-of-Sample Profit Factor** | 0.58 |
| **Net Realized P&L** | $-14.79 |
| **Total Fees Paid** | $49.98 |

---

## 2. Window-by-Window Out-of-Sample Breakdown

| Window | Model Ver | Out-of-Sample Period | Trades | Return | Win Rate | Sharpe | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W00 | `wf_w00_v1` | 2026-01-01 08:21 to 2026-01-01 10:00 | 2 | +0.07% | 50.0% | 29.17 | 0.30% |
| W01 | `wf_w01_v1` | 2026-01-01 10:01 to 2026-01-01 11:40 | 1 | -0.01% | 0.0% | -5.44 | 0.25% |
| W02 | `wf_w02_v1` | 2026-01-01 11:41 to 2026-01-01 13:20 | 2 | -0.21% | 50.0% | -85.51 | 0.30% |

---

## 3. Anti-Leakage Audit Summary

- Every window strictly satisfied: `max(T_train) < min(T_val) < min(T_test)`.
- Preprocessing and model parameter fitting occurred strictly within each window's training partition.
- Validation partition was used strictly for model parameter freezing.
- Zero out-of-sample test bars were observable prior to evaluation.
