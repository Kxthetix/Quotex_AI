# Phase 7 — Paper Trading Architecture

## 1. Overview and Core Invariant

Phase 7 creates a production-like paper-trading environment that operates on **live market data** while enforcing strictly **simulated execution**.

> [!CAUTION]
> **ABSOLUTE SAFETY INVARIANT**:
> Under NO circumstances can Phase 7 place a real-money broker or exchange order.
> There is NO code path connecting predictions, signals, or the paper execution engine to live trading credentials or withdrawal/deposit APIs. Even if production broker credentials or live URLs are configured, execution is intercepted and routed exclusively to the internal simulation adapter.

```
LIVE MARKET DATA (WebSocket)
       │
       ▼
CONNECTION MANAGER (Backoff, Heartbeat, Health)
       │
       ▼
DATA QUALITY & STALE DETECTOR (Circuit Breakers)
       │
       ▼
EVENT NORMALIZER
       ├───► ASYNC PERSISTENCE (Non-blocking Queue)
       ▼
IN-MEMORY MARKET STATE (Mark prices, Quotes, Depth)
       │
       ▼
LIVE CANDLE BUILDER (Multi-timeframe 1m, 5m, 15m, 1h, 4h, 1d)
       │
       ▼
FEATURE ENGINE (Deterministic, identical to Backtest)
       │
       ▼
LIVE INTELLIGENCE (Strict Model Version Lock & Anomaly Detection)
       │
       ▼
SIGNAL ENGINE (BUY / SELL / NO-TRADE Fusion)
       │
       ▼
INDEPENDENT RISK ENGINE (APPROVE / MODIFY / VETO Gate)
       │
       ▼
PAPER EXECUTION ENGINE (Spread, Slippage, Fees, Latency, Partials)
       │
       ▼
PAPER PORTFOLIO (Cash, Positions, Realized/Unrealized P&L)
       ├───► PREDICTION LEDGER (Immutable Records)
       ├───► TRADE LEDGER (Completed Cycles)
       ▼
OUTCOME TRACKER (MFE, MAE, Realized Return at Horizons)
       │
       ▼
SYSTEM HEALTH & AUDIT TELEMETRY (Latency p50/p95/p99, Alerting)
```

## 2. In-Memory Event-Driven Processing

The live processing pipeline is strictly **in-memory and event-driven**. Incoming events are never written to disk or database before decision-making:
1. Ticks and candles arrive via WebSocket and are validated in memory.
2. Market state, features, predictions, and signals are computed in microseconds.
3. Execution and portfolio updates occur synchronously in memory.
4. Downstream persistence occurs **asynchronously** via a dedicated persistence worker, ensuring that storage latency never stalls trading decisions.

## 3. Strict Layer Decoupling

- **Intelligence Layer**: Produces unconstrained directional predictions and probability distributions.
- **Signal Engine**: Fuses multiple intelligence streams into categorical `BUY`, `SELL`, or `NO_TRADE` actions with conviction scores.
- **Risk Engine**: Independent gate. Has full veto authority based on portfolio equity, exposure, daily loss, and drawdown limits.
- **Paper Execution**: Converts approved signals into simulated orders and fills under realistic market microstructure friction.
- **Ledgers**: Separates the **Prediction Ledger** (model forecast vs market reality) from the **Trade Ledger** (executed trades and P&L).
