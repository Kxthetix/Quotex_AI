# Controlled Model Promotion

## 1. Overview & Promotion Workflow

Promotion of a candidate model to production is strictly controlled and human/governance-gated:

```
CANDIDATE
   |
   v
VALIDATING
   |
   v
VALIDATED (Must pass all 10 validation gates)
   |
   v
APPROVAL_REQUIRED
   |
   v
APPROVED (Explicit sign-off by operator/governance)
   |
   v
PRODUCTION
```

## 2. Invariants & Safety Constraints

- Candidates **CANNOT** automatically transition to `PRODUCTION`.
- Attempting to promote an unvalidated model raises `PromotionError`.
- Direct manual state transitions to `PRODUCTION` via registry API are blocked with `PermissionError`.
- Previous production model is retired to `RETIRED` status, never deleted.
- Emits an immutable `PromotionRecord` in `promotion_history.json`.
