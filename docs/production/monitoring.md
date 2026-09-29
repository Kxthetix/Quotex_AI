# Production Monitoring and Metrics

## 1. Metrics Architecture

The production monitoring system provides thread-safe counters, gauges, and latency histograms for all operational domains:

### Mandatory Section 24 Metrics

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `market_events_total` | Counter | Total market events ingested. |
| `market_events_invalid_total` | Counter | Market events rejected by validation breaker. |
| `market_data_disconnects_total` | Counter | Total socket disconnection occurrences. |
| `market_data_reconnects_total` | Counter | Successful socket reconnections. |
| `predictions_total` | Counter | Total directional and quality predictions generated. |
| `signals_total` | Counter | Total actionable BUY/SELL/NO-TRADE signals emitted. |
| `orders_total` | Counter | Total paper execution orders placed. |
| `fills_total` | Counter | Total simulated order fills completed. |
| `trades_total` | Counter | Total completed round-trip trades. |
| `errors_total` | Counter | Total system runtime exceptions. |
| `circuit_breakers_total` | Counter | Times risk/data circuit breaker tripped. |
| `database_errors_total` | Counter | PostgreSQL persistence failures. |
| `redis_errors_total` | Counter | Redis connection or query failures. |
| `processing_latency` | Histogram | End-to-end pipeline latency (seconds). |
| `prediction_latency` | Histogram | Feature vector to model inference latency (seconds). |
| `signal_latency` | Histogram | Signal fusion decision latency (seconds). |
| `risk_latency` | Histogram | Risk engine check and sizing latency (seconds). |
| `execution_latency` | Histogram | Simulated fill placement latency (seconds). |

### System & State Gauges

- `system_cpu_percent`: Host CPU utilization.
- `system_memory_mb`: Process Resident Set Size (RSS) in megabytes.
- `system_disk_percent`: Disk capacity percentage.
- `active_exposure`: Current open dollar risk exposure.
- `current_drawdown`: Current portfolio peak-to-trough drawdown percentage.
- `data_freshness_seconds`: Age of the most recently ingested market tick.
