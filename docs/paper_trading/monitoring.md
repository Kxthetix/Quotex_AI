# Monitoring, Telemetry & Alerting

## 1. System Health States

The master system health is synthesized from all component statuses:

| State | Condition |
| :--- | :--- |
| `HEALTHY` | All feeds active, latencies nominal, no circuit breakers tripped. |
| `DEGRADED` | Stale market data, reconnection in progress, or risk breaker active. |
| `BLOCKED` | Data-Quality Breaker tripped or severe Model Anomaly detected. |
| `FAILED` | Connection manager exhausted retries or critical process crash. |

## 2. Latency Monitoring (Section 31)

Latency percentiles ($p50, p95, p99$) are continuously maintained across 9 tracks:
1. `receive_latency`
2. `normalization_latency`
3. `candle_latency`
4. `feature_latency`
5. `inference_latency`
6. `signal_latency`
7. `risk_latency`
8. `execution_latency`
9. `total_pipeline_latency`

## 3. Tiered Alerting Architecture (Section 35)

Structured `MonitoringEvent` objects are categorized into three severity levels:

- **`CRITICAL`**:
  - `DATA_QUALITY`: Corrupt ticks or OHLC inconsistencies.
  - `STALE_DATA`: Data interval exceeded threshold.
  - `MODEL_ANOMALY`: NaN, out-of-bounds probability, version mismatch.
  - `CIRCUIT_BREAKER`: Risk limit breached (max drawdown, daily loss).
- **`WARNING`**:
  - Elevated processing latency.
  - Connection retries or transient reconnect.
  - Missing prediction evaluation horizon.
- **`INFO`**:
  - Session start / session stop.
  - Model loaded under version lock.
  - Prediction evaluated.
  - Trade completed.
