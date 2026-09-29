# Production Alerting System

## 1. Alert Tiers

Alerts are routed into three distinct operational severities:

### CRITICAL
Immediate action required; automated trading is blocked or compromised:
- Market data unavailable or socket disconnect > 15s.
- Stale data detected (> 65s).
- Database or Redis connection failure.
- Model service unavailable or returning NaNs.
- Risk engine circuit breaker tripped.
- Tamper detection alert on audit trail or backup archive.
- Security authorization violation or credential leak.

### WARNING
Component degraded; live trading may proceed with caution:
- Repeated WebSocket reconnect cycles (>= 3 in 5 minutes).
- Pipeline latency p95 > 150ms.
- Free disk space < 15%.
- Scheduled backup failed 5-point verification.
- Feature/prediction drift detected (PSI > 0.20).

### INFO
Standard operational milestone logged for observability:
- Service started or clean shutdown completed.
- Market data feed successfully reconnected and gap-backfilled.
- Scheduled backup successfully verified.
- Candidate model promoted or rolled back by authorized user.

## 2. Cooldown Throttling and Storm Prevention

To prevent notification storms during cascading failure events:
- Deduplication key: `{severity}:{component}:{title}`.
- Default cooldown window: 60.0 seconds.
- Duplicate alerts matching the key within the window are suppressed.
