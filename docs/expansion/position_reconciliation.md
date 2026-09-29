# Position Reconciliation & Safety Interventions

## 1. Discrepancy Detection (Section 34, 58)

The system compares internal portfolio ledger positions against actual broker balances:
$$\Delta = |Q_{\text{internal}} - Q_{\text{provider}}|$$

If $\Delta > 10^{-5}$:
1. Flag `reconciliation_required` is set to `True`.
2. Operational state transitions to `POSITION_RECONCILIATION_REQUIRED`.
3. All new order placement for the affected instrument is immediately BLOCKED.
4. A CRITICAL audit record is generated with full delta details.

## 2. Return-to-Service Protocol

Trading cannot resume automatically after a position discrepancy:
1. Operator investigates root cause (e.g. external manual trade or missed fill callback).
2. Ledger or broker balance is reconciled.
3. Authorized operator issues signed `resolve_discrepancy()` command.
4. System readiness evaluator verifies zero discrepancies before unlocking order execution.
