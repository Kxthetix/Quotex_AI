# Production Incident Runbooks

The system defines Standard Operating Procedures (SOP) across all 12 mandatory failure modes:

| ID | Runbook Title | Primary Component | Immediate Safety Action |
| :--- | :--- | :--- | :--- |
| **RB-01** | WebSocket Disconnection Outage | `MarketData` | Block new signal decisions; halt order placement. |
| **RB-02** | Stale Market Data Detection | `MarketData` | Trip stale-data breaker; set TradingState to BLOCKED. |
| **RB-03** | PostgreSQL / TimescaleDB Outage | `Database` | Divert writes to bounded `PersistenceBuffer`; block live orders. |
| **RB-04** | Redis Cache Outage | `Redis` | Switch to direct feature compute; disable pub/sub fanout. |
| **RB-05** | Model Service Inference Failure | `ModelService` | Fallback to NO-TRADE default; block order placement. |
| **RB-06** | Model Artifact Corruption | `ModelRegistry` | Refuse corrupted weights; retrieve verified backup. |
| **RB-07** | High Pipeline Latency | `Pipeline` | Rate-limit event processing; purge stale incoming queues. |
| **RB-08** | Disk Approaching Capacity | `Infrastructure` | Pause non-essential logging; purge archives > 30 days. |
| **RB-09** | Backup Execution/Verification Failure | `BackupManager` | Fire WARNING alert; retain previous successful backup. |
| **RB-10** | Circuit Breaker Tripped | `RiskEngine` | Block order generation; cancel pending unfilled orders. |
| **RB-11** | Unexpected Model Behavior / Drift | `LearningLoop` | De-weight model signals; switch to conservative mode. |
| **RB-12** | Security Incident / Unauthorized Access | `Security` | Revoke compromised credentials; lock admin endpoints. |

Each runbook details 6 steps:
1. Detection mechanism
2. Immediate safety action
3. Diagnostic procedure
4. Recovery protocol
5. Validation checks
6. Return-to-service criteria
