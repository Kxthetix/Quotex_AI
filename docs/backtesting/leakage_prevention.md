# Phase 5 — Anti-Leakage Invariants and Critical Proofs

## 1. Principles of Anti-Leakage Simulation

In quantitative trading, lookahead leakage produces deceptively optimistic backtest results that instantly fail in live trading. Phase 5 enforces three foundational anti-leakage barriers:

1. **Chronological Candle Completion Horizon:** A candle spanning $[T_{start}, T_{end}]$ only becomes observable at $T \ge T_{end}$. Decisions cannot use $T_{end}$ close prices prior to candle closure.
2. **Execution Latency Barrier:** An order submitted at $T_{submit}$ cannot execute prior to $T_{submit} + \Delta t_{latency}$. Fill prices must come from market data available at $T_{exec}$, never at decision time.
3. **Partition Isolation Invariant:** Model training and preprocessing parameter fitting are strictly confined to past partitions.

---

## 2. Mandatory Verification Tests

### Section 57: Critical Leakage Invariant Test
Verified in `tests/unit/backtesting/test_backtest_leakage_critical.py`:
1. Run backtest on historical series through timestamp $T$.
2. Record all orders, fills, and decisions up to $T$.
3. Append historical candles occurring after $T$.
4. Replay backtest up to $T$ using the expanded dataset.
5. **Assertion:** Every trade record, entry price, exit price, and audit log entry up to $T$ remains bit-for-bit identical.

### Section 58: Critical Execution Latency Test
Verified in `tests/unit/backtesting/test_backtest_leakage_critical.py`:
1. Configure execution latency $\Delta t = 60\text{s}$.
2. Generate trading signal on Candle 1 (ending at $10:01$).
3. **Assertion:** $T_{signal} (10:01) < T_{execution} (10:02)$.
4. **Assertion:** The execution fill price is taken from Candle 2 (at $10:02$), proving future prices are not retroactively filled.

### Section 43: Walk-Forward Leakage Test
Verified in `tests/unit/backtesting/test_walk_forward.py`:
- In every rolling walk-forward window:
  $$\max(T_{train}) < \min(T_{val}) < \min(T_{test})$$
  guaranteeing zero out-of-sample data leakage into model training or calibration.
