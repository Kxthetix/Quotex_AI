# Phase 5 — Backtesting Engine Architecture

## 1. Objective and Architectural Overview

The Phase 5 Backtesting Engine simulates the behavior of the Trading Intelligence System against historical market data under realistic market microstructure and execution constraints.

The backtester is **event-driven** and executes strictly **chronologically**:

```
                 HISTORICAL MARKET DATA
                           |
                           v
                    EVENT REPLAY
                           |
                           v
                    MARKET STATE
                           |
                           v
                    FEATURE ENGINE
                           |
                           v
                 INTELLIGENCE MODELS
                           |
                           v
                    SIGNAL INPUT
                           |
                           v
                     ORDER ENGINE
                           |
                           v
                  EXECUTION SIMULATOR
                           |
                           v
                      POSITION
                           |
                           v
                      PORTFOLIO
                           |
                           v
                    PERFORMANCE
```

## 2. Event Stream and Hierarchy

Every event in the simulation derives from `BacktestEvent` and carries an explicit UTC `timestamp`:

1. `CandleEvent`: Emitted strictly at the candle's completion time (`end_time`). A candle spanning `[start_time, end_time]` can only influence decisions at or after `end_time`.
2. `FeatureUpdateEvent`: Emitted after feature transformations have completed for the closed candle.
3. `SignalEvent`: Standardized intelligence signal containing directional probabilities, expected returns, volatility estimates, regime classification, and candidate action (`ENTER_LONG`, `ENTER_SHORT`, `EXIT`, `HOLD`).
4. `OrderEvent`: Emitted when an order is created, validated, and submitted.
5. `ExecutionEvent`: Emitted when an order fill occurs, capturing fill price, quantity, fees, spread cost, and slippage cost.
6. `PositionUpdateEvent`: Emitted when open position quantities, entry prices, or unrealized P&L change.
7. `PortfolioUpdateEvent`: Emitted on mark-to-market updates and cash balance changes.

## 3. Historical Replay Engine

The `HistoricalReplayEngine` (`backtesting/replay.py`):
- Sorts candles strictly chronologically by `(end_time, symbol)`.
- Validates temporal monotonicity: rejects or warns on out-of-order records.
- Detects market data gaps and corrupt candle geometry (e.g. `low > high`, negative volume).
- Yields `CandleEvent` instances one-by-one to prevent future data inspection.

## 4. Anti-Leakage Invariants

- **Decision Time Horizon:** At timestamp $T$, only data available at $\le T$ is accessible.
- **Latency Separation:** Signal timestamp $T_{signal} < T_{execution}$ whenever latency is configured.
- **Deterministic Replay:** Given identical datasets, configurations, and random seeds, backtest results are bit-for-bit reproducible.
