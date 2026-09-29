# Circuit Breakers Specification

Phase 7 implements four distinct circuit breakers to protect against data corruption, network failure, model degradation, and portfolio losses:

## 1. Data-Quality Circuit Breaker

- **Source**: `DataQualityEngine` (`data/validation/engine.py`).
- **Triggers**:
  - Consecutive timestamp violations (clock drift > 5.0s, future timestamps).
  - Outlier price jump > 20% in a single tick.
  - Critical OHLC inconsistencies ($\text{High} < \text{Low}$, $\text{Close} > \text{High}$).
- **Action**:
  - Sets system status to `BLOCKED`.
  - Immediately drops corrupt events.
  - Halts live signal generation.
  - Requires inspection or manual reset via `/api/v1/market/breaker/reset`.

## 2. Stale-Data Circuit Breaker

- **Source**: `StaleDataDetector` (`paper_trading/stale_detector.py`).
- **Triggers**:
  - Event receive interval exceeds `stale_threshold_seconds` (default: 15.0s).
- **Action**:
  - Sets system status to `DEGRADED`.
  - Halts all new signal generation and new paper orders.
  - Resumes automatically when fresh, valid market data is received.

## 3. Risk Circuit Breaker

- **Source**: `RiskEngine` (`paper_trading/risk_engine.py`).
- **Triggers**:
  - Drawdown $\ge 10\%$ from peak equity.
  - Daily loss $\ge 3\%$ of total equity.
  - Trade frequency exceeds limit (e.g. > 20 trades/hour).
- **Action**:
  - Vetoes all new opening/increasing orders.
  - Permits risk-reducing or closing orders to de-risk.
  - Preserves current portfolio state and records critical monitoring event.

## 4. Model Anomaly Circuit Breaker

- **Source**: `ModelAnomalyDetector` (`paper_trading/monitoring.py`).
- **Triggers**:
  - Prediction confidence is `NaN` or infinite.
  - Confidence $< 0.0$ or $> 1.0$.
  - Impossible expected forward return ($|R| > 100\%$).
  - Model version or feature version mismatch.
- **Action**:
  - Immediately neutralizes prediction to `NO_TRADE`.
  - Sets system state to `BLOCKED`.
  - Prohibits order placement.
  - **Does NOT replace the model automatically**. Requires human operator review.
