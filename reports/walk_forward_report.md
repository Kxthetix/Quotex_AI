# Rolling Walk-Forward Validation Report

**Windows Evaluated:** 3
**Leakage Invariant Status:** **PASSED (Strict Monotonic Separation)**
**Window Sizing:** Train=300 bars, Val=100 bars, Test=100 bars, Step=100 bars

---

## 1. Aggregate Out-Of-Sample Performance

| Metric | Aggregate Out-of-Sample Result |
| :--- | :--- |
| **Combined OOS Trades** | 10 |
| **Combined OOS Return** | **+9.29%** |
| **Out-of-Sample Sharpe Ratio** | **34.08** |
| **Out-of-Sample Sortino Ratio** | 25.70 |
| **Out-of-Sample Max Drawdown** | 8.07% |
| **Out-of-Sample Win Rate** | 30.0% |
| **Out-of-Sample Profit Factor** | 1.71 |
| **Net Realized P&L** | $+929.02 |
| **Total Fees Paid** | $70.40 |

---

## 2. Window-by-Window Out-of-Sample Breakdown

| Window | Model Ver | Out-of-Sample Period | Trades | Return | Win Rate | Sharpe | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W00 | `wf_w00_v1` | 2020-03-17 17:59 to 2020-03-21 20:59 | 3 | +10.91% | 66.7% | 79.07 | 7.50% |
| W01 | `wf_w01_v1` | 2020-03-21 21:59 to 2020-03-26 00:59 | 3 | +2.97% | 33.3% | 40.42 | 4.17% |
| W02 | `wf_w02_v1` | 2020-03-26 01:59 to 2020-03-30 04:59 | 4 | -4.31% | 0.0% | -124.96 | 5.00% |

---

## 3. Anti-Leakage Audit Summary

- Every window strictly satisfied: `max(T_train) < min(T_val) < min(T_test)`.
- Preprocessing and model parameter fitting occurred strictly within each window's training partition.
- Validation partition was used strictly for model parameter freezing.
- Zero out-of-sample test bars were observable prior to evaluation.
