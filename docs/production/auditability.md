# Audit Trail and Causality Correlation

## 1. End-to-End Causality Chain

To investigate "Why did this trade occur?", every operation is tagged with correlation IDs propagating across the pipeline:

```
market_event_id
       │
       ▼
prediction_id
       │
       ▼
signal_id
       │
       ▼
risk_decision_id
       │
       ▼
order_id
       │
       ▼
fill_id
       │
       ▼
trade_id
```

Each link is held in `CorrelationChain` via thread-safe `contextvars`.

## 2. Immutable Hash-Chained Audit Trail

The audit log (`data/audit/audit_trail.jsonl`) uses cryptographic SHA-256 hash chaining:
- Each record calculates:
  $$\text{record\_hash} = \text{SHA256}(\text{record\_id} + \text{timestamp} + \text{actor} + \text{action} + \text{prev\_hash})$$
- Genesis record begins with `prev_hash = "GENESIS"`.
- Any external modification, truncation, or insertion breaks the chain and is flagged immediately by `AuditLogger.verify_integrity()`.

## 3. Idempotency Protection

To prevent duplicate trades or double mutations during network retries:
- `IdempotencyManager` tracks deterministically computed operation hashes.
- In-progress locking prevents race conditions.
- Re-executing an already completed key returns the cached result without duplicate execution.
