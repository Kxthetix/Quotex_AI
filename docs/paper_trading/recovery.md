# Restart Recovery & State Preservation

## 1. Recovery Architecture

The `SessionRecoveryEngine` ensures that paper trading sessions can be safely stopped, checkpointed, and resumed without state loss or duplicate records:

```
               RUNNING SESSION
                      │
                      ▼ (Save Checkpoint)
          session_{id}_checkpoint.json
                      │
                      ├──► Session metadata & versions
                      ├──► PaperPortfolio (cash, equity, peak)
                      ├──► Open positions
                      ├──► Paper Trade Ledger
                      ├──► Prediction Ledger & pending evaluations
                      └──► Order Engine state
                      │
                      ▼ (Restart)
             RESTORED SESSION
```

## 2. Idempotent Deduplication Invariants

Upon system restart, duplicate records could cause catastrophic distortion in accounting and model evaluation. The recovery engine enforces:

1. **Trade Idempotence**: Existing `trade_id` records in the Trade Ledger are deduplicated; replay cannot record the same closed trade twice.
2. **Prediction Idempotence**: Existing `prediction_id` entries are checked before recording. Original predictions are never overwritten.
3. **Pending Evaluation Queue**: Predictions whose evaluation horizon had not elapsed prior to shutdown are restored to `PENDING` status and evaluated once current market prices become available after restart.
4. **Portfolio Balances**: Cash, equity, peak equity, and open positions are restored to the exact millisecond of the checkpoint.
