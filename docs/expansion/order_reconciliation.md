# Order Reconciliation Protocol

## 1. Internal vs External Order Tracking

The `ExecutionReconciliationEngine` reconciles order status:
- Internal order state (in `ExecutionCoordinator`).
- External broker state (queried from `ExecutionProvider.query_order`).

## 2. Discrepancy Resolution

If an order is marked `SUBMITTED` internally but reported as `REJECTED` or `CANCELLED` by the exchange:
1. The internal order transitions to matching terminal state.
2. A reconciliation log entry is appended to the audit trail.
3. If an unrecognized external fill is discovered, a CRITICAL alert is triggered.
