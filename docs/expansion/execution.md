# Execution Integration & Safety Pipeline

## 1. Execution Architecture

The pipeline enforces strict separation of concerns:
```
Prediction Engine
       │
       ▼
 Signal Engine
       │
       ▼
  Risk Engine   ◄─── (MANDATORY VETO LAYER)
       │
       ▼
Execution Policy
       │
       ▼
Execution Provider Adapter
```

Prediction models NEVER submit orders directly.

## 2. Order Lifecycle States

Orders progress through an explicit state machine:
- `CREATED`: Constructed by ExecutionCoordinator.
- `VALIDATED`: Ticker, minimum lot size, and mode verified.
- `SUBMITTED`: Dispatched to broker/exchange session.
- `ACKNOWLEDGED`: Order ID assigned by provider.
- `PARTIALLY_FILLED`: Partial fill executed.
- `FILLED`: Fully executed.
- `CANCEL_REQUESTED` / `CANCELLED`: User or system cancellation.
- `REJECTED`: Vetoed by Risk Engine or broker checks.
- `EXPIRED` / `ERROR`: Time-in-force expiry or connection failure.

## 3. Execution Modes & Safety Guard

- Supported modes: `BACKTEST`, `PAPER`, `SHADOW`, `SIMULATED`, `LIVE`.
- ModeGuard strictly blocks `LIVE` execution unless `ALLOW_LIVE_REAL_MONEY_TRADING=true` is explicitly set in environment variables.
