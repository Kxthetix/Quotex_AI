# Expansion Monitoring and Telemetry

## 1. New Expansion Metrics

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `orderbook_updates_total` | Counter | Total Level 2 incremental updates processed. |
| `orderbook_sequence_gaps_total` | Counter | Sequence gaps detected requiring snapshot rebuilds. |
| `provider_failovers_total` | Counter | Controlled failover occurrences. |
| `provider_divergence_events_total` | Counter | Cross-provider price discrepancies $> 0.5\%$. |
| `sequence_inference_latency` | Histogram | Inference latency for TemporalSequenceModel (seconds). |
| `position_discrepancies_total` | Counter | Times position reconciliation mismatch was triggered. |
| `multi_asset_risk_vetoes_total` | Counter | Multi-asset concentration or leverage vetoes. |

## 2. Telemetry Gauges

- `active_provider_priority`: Currently active feed (1=Primary, 2=Secondary, 3=Tertiary).
- `portfolio_concentration_pct`: Max single-asset concentration percentage.
- `portfolio_aggregate_leverage`: Current total portfolio leverage ratio.
