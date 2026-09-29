# Shadow Model Evaluation

## 1. Overview & Mechanics

Shadow model evaluation enables side-by-side comparison of candidate models against the active production model using identical live observations without taking live risk:

- **Production Model**: ACTIVE. Drives paper-trading signals, risk evaluation, and order placement.
- **Candidate Model**: SHADOW. Receives identical normalized market features, executes inference, and records predictions to the shadow ledger.
- **Shadow Model Ledger**: Stores timestamp, production action/confidence, candidate action/confidence, and subsequent realized outcome.

## 2. Advantages

- Zero execution risk: Candidate models produce zero orders.
- Direct comparability: Side-by-side evaluation on identical real-time market data.
- Ground-truth evaluation: Outcomes match actual price evolution over time.
