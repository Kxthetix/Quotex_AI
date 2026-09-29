# Validation Methodology & Backtest Comparison

## 1. Paper Validation Period Criteria (Section 45)

Before paper-trading results can be submitted to the Model Auditor in Phase 8, the session must satisfy configurable validation thresholds:

```python
@dataclass
class PaperValidationCriteria:
    minimum_paper_days: int = 7
    minimum_predictions: int = 100
    minimum_trades: int = 20
    minimum_data_coverage_pct: float = 80.0
```

The `PaperTradingReporter.evaluate_validation_period()` function evaluates these criteria and produces an audit assessment.

## 2. No Automatic Promotion (Section 46)

> [!IMPORTANT]
> Paper trading performance NEVER automatically promotes a model to production.

The governance hierarchy is strictly staged:
```
PAPER RESULTS
     │
     ▼
MODEL AUDITOR (Phase 8)
     │
     ▼
VALIDATION REVIEW
     │
     ▼
GOVERNANCE GATE
     │
     ▼
PROMOTE OR REJECT
```
Phase 7 only produces verifiable empirical evidence; promotion authority belongs exclusively to the governance loop in Phase 8.

## 3. Backtest vs Paper Comparison (Section 43)

The reporter compares live paper trading against Phase 5 backtest baselines:
- **Trade Frequency**: Identifies whether live latency or risk filters throttled trade frequency relative to backtest.
- **Win Rate & P&L**: Detects if market regime shifts or execution slippage eroded expected edge.
- **Slippage & Fees**: Confirms simulated frictions match or exceed empirical backtest assumptions.
- **Governance Flag**: If live win rate diverges from backtest by $> 15\%$, a `FLAG_FOR_INVESTIGATION` alert is generated.
