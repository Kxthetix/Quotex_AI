# Production Troubleshooting Guide

## 1. Quick Diagnostic Flowchart

```
ISSUE OBSERVED
      │
      ├─► Run Health & Readiness Probe:
      │     python -c "from production.health.checker import HealthChecker; print(HealthChecker().check_all())"
      │
      ├─► Check Audit Trail Integrity:
      │     python -c "from production.audit.audit_logger import AuditLogger; print(AuditLogger().verify_integrity())"
      │
      ├─► Check Active Critical Alerts:
      │     python -c "from production.monitoring.alerting import AlertManager; print(AlertManager().get_active_critical_alerts())"
      │
      └─► Check Unresolved Incidents:
            python -c "from production.incidents.incident_manager import IncidentManager; print(IncidentManager().get_open_incidents())"
```

## 2. Common Symptoms and Resolutions

### Symptom: `SystemReadinessReport.is_ready == False`
- **Cause**: At least one mandatory component (`MarketData`, `Database`, `Redis`, `ModelService`, `RiskEngine`) failed health probing.
- **Resolution**: Inspect `report.blocking_issues`. If `Database` failed, check PostgreSQL container logs and connection limits. If `MarketData` failed, verify exchange endpoint connectivity.

### Symptom: `PersistenceBuffer` Dropping Records
- **Cause**: Database outage exceeded memory buffering capacity (default 5,000 records).
- **Resolution**: Check database server status immediately. Once DB is online, trigger `buffer.flush(write_fn)` to persist buffered records.

### Symptom: Stale Data Breaker Tripped
- **Cause**: Last closed candle age exceeded 65.0 seconds.
- **Resolution**: Check system clock synchronization (NTP) and exchange WebSocket connection. Request REST backfill if bars were dropped. Reset breaker with `OPERATOR` credentials.

### Symptom: Backup Verification Mismatch
- **Cause**: Archive file hash does not match `.manifest.json`.
- **Resolution**: Check if backup was truncated or partially overwritten. Rerun backup command for the affected domain and verify storage volume health.
