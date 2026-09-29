# Phase 8 — Learning Loop Architecture

## 1. System Overview

The Learning Loop establishes a controlled machine-learning lifecycle around the Trading Intelligence System. It continuously collects empirical evidence from historical data, backtests, walk-forward simulations, paper trading, prediction ledgers, model metrics, feature distributions, market regimes, and execution outcomes to determine whether the active production model remains valid.

```
                    HISTORICAL DATA
                          |
                          v
                    DATA QUALITY
                          |
                          v
                    FEATURE DATA
                          |
             +------------+------------+
             |                         |
             v                         v
       CURRENT MODEL             CANDIDATE DATA
             |                         |
             v                         v
       PREDICTIONS                RETRAINING
             |                         |
             v                         v
       OUTCOME LEDGER           CANDIDATE MODEL
             |                         |
             +------------+------------+
                          |
                          v
                    MODEL AUDITOR
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
           DRIFT      PERFORMANCE   STABILITY
              |           |           |
              +-----------+-----------+
                          |
                          v
                   VALIDATION GATES
                          |
                          v
                    MODEL REGISTRY
                          |
                          v
                  HUMAN/CONTROLLED
                     PROMOTION
                          |
                          v
                   PRODUCTION MODEL
```

## 2. Core Safety Invariants

1. **No Online Learning**: Production model weights are NEVER updated directly from individual live observations.
2. **Immutable Production Models**: Active production models cannot be overwritten or silently replaced.
3. **Candidate Isolation**: Candidates live in dedicated storage directories (`data/models/candidates/`) completely separated from production (`data/models/production/`).
4. **Controlled Promotion Gatekeeper**: Candidates require passing all 10 validation gates and receiving explicit approval before promotion to production.
5. **Deterministic Reproducibility**: Fixed random seeds, exact feature schemas, dataset versions, and code versions guarantee reproducible candidate training.
6. **No Real-Money Trading**: Execution simulation remains paper-only.
