# Market Data Gap Recovery

## 1. Gap Detection Logic

The `MarketDataGapRecovery` engine tracks the timestamp of the last verified closed candle for each active trading pair.

When a reconnection occurs at `t_current`:
$$\Delta t = t_{\text{current}} - t_{\text{last\_seen}}$$

If $\Delta t > 65.0\text{ seconds}$ (for 1-minute timeframes):
- Missing bar count estimate: $N = \lfloor \Delta t / 60 \rfloor$.
- An operational gap event is recorded.

## 2. Historical Backfill Procedure

1. The gap recovery engine invokes the registered `backfill_fn(symbol, t_last_seen, t_current)`.
2. REST historical candle endpoints are queried for the missing interval $[t_{\text{start}}, t_{\text{end}}]$.
3. Backfilled bars are validated:
   - Chronological sequence verification.
   - Non-zero OHLC prices ($L \le O, C \le H$).
   - Absence of duplicate timestamps.
4. If backfill returns valid candles matching the missing count:
   - Gap is flagged as `is_recovered = True`.
   - The symbol block is cleared.
   - Live trading resumes.

## 3. Unrecoverable Gaps

If historical REST backfill fails or is unavailable:
- The symbol is added to `_blocked_symbols`.
- Signal generation and trading decisions for that symbol remain BLOCKED.
- A CRITICAL alert is triggered for operator intervention.
