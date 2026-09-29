# Phase 10 Expansion Troubleshooting

## 1. Common Operational Issues

### Issue: Order Book Status = `GAP_DETECTED`
- **Cause**: WebSocket dropped an incremental packet or sequence number skipped.
- **Remedy**: Request fresh full Level 2 snapshot via `provider.fetch_orderbook_snapshot()`. Pass to `tracker.apply_snapshot()`.

### Issue: `reconciliation_required = True`
- **Cause**: Broker position quantity does not match internal ledger balance ($\Delta > 10^{-5}$).
- **Remedy**: Review external broker fills. If valid manual deposit/withdrawal occurred, execute `engine.resolve_discrepancy(symbol, actor, notes)`.

### Issue: `RiskVetoError` on Multi-Asset Trade
- **Cause**: Single-asset concentration exceeded (default 35%) or aggregate portfolio leverage breached 1.0x.
- **Remedy**: Reduce order sizing or adjust risk configuration with `ADMIN` role sign-off.

### Issue: `TenantIsolationViolationError`
- **Cause**: User attempted to view or submit orders on a portfolio not owned by their account or tenant.
- **Remedy**: Verify user session credentials and target portfolio ID mapping.
