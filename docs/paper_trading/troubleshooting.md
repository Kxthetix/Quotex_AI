# Troubleshooting & Operational Guide

## 1. Common Operational Scenarios

### A. Data-Quality Circuit Breaker Active (`BLOCKED`)
- **Symptom**: System state is `BLOCKED`. Health endpoint shows `data_quality_breaker_tripped: true`.
- **Cause**: Incoming ticks had invalid timestamps, prices leaped $> 20\%$, or OHLC was inverted.
- **Resolution**:
  1. Inspect recent incidents via `GET /api/v1/market/quality/incidents`.
  2. Verify exchange feed integrity or WebSocket connectivity.
  3. Reset the breaker manually:
     ```bash
     curl -X POST http://localhost:8000/api/v1/market/breaker/reset
     ```

### B. Stale Data Detected (`DEGRADED`)
- **Symptom**: System state is `DEGRADED`. Health reports `stale_data: true`.
- **Cause**: No ticks or candles received for $> 15.0$ seconds.
- **Resolution**:
  - The ConnectionManager automatically attempts exponential backoff reconnection.
  - Signal generation and orders are paused automatically to protect capital.
  - Fresh ticks will automatically restore system state to `HEALTHY`.

### C. Model Anomaly Halting Decisions
- **Symptom**: Model anomaly reported; signals neutralized to `NO_TRADE`.
- **Cause**: Model inference produced NaN, negative probability, or version mismatch.
- **Resolution**:
  1. Inspect `model_anomaly_reason` in health report.
  2. Verify that features being passed match the locked model version.
  3. Do NOT hot-swap the model in paper trading. If retraining is needed, run controlled governance training in Phase 8.

### D. Risk Engine Circuit Breaker Tripped
- **Symptom**: Signal engine generates `BUY`/`SELL`, but risk engine returns `decision: VETO`.
- **Cause**: Drawdown exceeded 10%, daily loss exceeded 3%, or hourly trades exceeded 20.
- **Resolution**:
  - Review open positions and market conditions.
  - Risk-reducing orders (closing long/short) remain permitted to de-risk.
  - Breaker automatically resets daily counters at 00:00 UTC.

## 2. CLI and Operational Commands

- **Run Paper Trading Unit Tests**:
  ```bash
  .venv\Scripts\pytest tests/unit/paper_trading/
  ```

- **Run Full System Test Suite**:
  ```bash
  .venv\Scripts\pytest
  ```

- **Inspect Live API Endpoints**:
  ```bash
  curl http://localhost:8000/api/v1/paper/status
  curl http://localhost:8000/api/v1/paper/portfolio
  curl http://localhost:8000/api/v1/paper/signals
  curl http://localhost:8000/api/v1/paper/predictions
  curl http://localhost:8000/api/v1/paper/trades
  curl http://localhost:8000/api/v1/paper/health
  curl http://localhost:8000/api/v1/paper/report
  ```
