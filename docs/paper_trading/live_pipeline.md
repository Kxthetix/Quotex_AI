# Live Pipeline Specification

## 1. Pipeline Stages

The Phase 7 live pipeline converts incoming market events into simulated portfolio fills across 8 discrete steps:

1. **Ingestion & Connection Management**:
   - Manages WebSocket stream lifecycles (`CONNECTING`, `CONNECTED`, `DEGRADED`, `RECONNECTING`, `FAILED`).
   - Exponential backoff with random jitter on disconnects.
   - Heartbeat handling and disconnection callbacks.

2. **Validation & Data Quality Engine**:
   - Clock drift and future timestamp check.
   - Event deduplication via rolling sliding window cache.
   - OHLC consistency and non-negative volume validation.
   - Price outlier jump detection.
   - Data-Quality Circuit Breaker trips immediately upon critical data corruption.

3. **Stale Data Protection**:
   - Monitors arrival interval per symbol.
   - If interval exceeds `stale_threshold_seconds` (default: 15.0s), system marks state `DEGRADED` and immediately suppresses new signals and orders until valid stream resumes.

4. **Event Normalization & In-Memory State**:
   - Converts raw provider payloads into canonical `NormalizedTrade` and `Candle` entities.
   - Maintains real-time in-memory `MarketState`: mark price, bid, ask, volume, and active/closed candles.

5. **Candle Building**:
   - Aggregates trades into standard timeframes (1m, 5m, 15m, 1h, 4h, 1d).
   - Late event correction policy ensures auditable handling of delayed arrivals.

6. **Feature Calculation**:
   - Reuses Phase 3 `FeatureEngine`.
   - Generates exact feature vectors using the identical calculations as historical backtests.

7. **Intelligence & Signal Generation**:
   - Locked production models generate directional probabilities and expected returns.
   - Model anomaly detection checks for NaN, probability bounds, and distribution collapse.
   - Phase 6 `SignalEngine` fuses evidence into `BUY`, `SELL`, or `NO_TRADE` with conviction score.

8. **Risk Gating & Paper Execution**:
   - Independent `RiskEngine` evaluates signal against portfolio constraints.
   - `PaperExecutionEngine` simulates realistic market frictions: spread, slippage, maker/taker fees, and latency.
   - `PaperPortfolioManager` updates cash, positions, and closed trade ledger.

## 2. Latency Measurement Points

Latency is measured at each stage boundary and recorded into percentiles ($p50, p95, p99$):
- $T_{\text{event}}$: Timestamp on exchange
- $T_{\text{receive}}$: Local reception timestamp
- $T_{\text{normalize}}$: Completion of normalization
- $T_{\text{candle}}$: In-memory candle accumulation
- $T_{\text{feature}}$: Feature calculation completion
- $T_{\text{inference}}$: Model prediction generated
- $T_{\text{signal}}$: Signal Engine fusion completed
- $T_{\text{risk}}$: Risk evaluation approved/modified/vetoed
- $T_{\text{exec}}$: Simulated fill execution timestamp
