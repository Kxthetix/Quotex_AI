# Prediction Ledger

## 1. Immutability Invariant

The **Prediction Ledger** is an append-only historical record of every prediction produced by the intelligence layer.

> [!IMPORTANT]
> Predictions are immutable. Original forecast attributes (timestamp, confidence, signal action, expected return, stop, and target prices) are **never overwritten**. When evaluation horizons elapse, outcome metrics are appended to the existing record.

## 2. PredictionRecord Schema

```python
@dataclass
class PredictionRecord:
    prediction_id: str
    symbol: str
    timestamp: datetime
    timeframe: str
    model_version: str
    feature_version: str
    signal_version: str
    signal_action: str          # BUY, SELL, NO_TRADE
    confidence: float
    regime: str
    setup: str
    expected_return: float
    expected_volatility: float
    expected_duration_bars: int
    entry_price: float
    stop_price: float
    target_price: float
    evaluation_horizon_seconds: float
    # Evaluation fields (appended later)
    status: PredictionStatus    # PENDING, EVALUATED, EXPIRED
    evaluation_time: datetime | None
    realized_return: float | None
    mfe: float | None           # Maximum Favorable Excursion
    mae: float | None           # Maximum Adverse Excursion
    actual_outcome: str | None  # CORRECT, INCORRECT, NEUTRAL
    prediction_error: float | None
    metadata: dict[str, Any]
```

## 3. Querying the Ledger

The ledger provides programmatic queries:
- `get_pending_predictions()`: Active predictions awaiting horizon completion.
- `get_evaluated_predictions()`: Completed predictions with ground-truth outcomes.
- `count_by_action()`: Signal action breakdown (`BUY`, `SELL`, `NO_TRADE`).
- `count_by_outcome()`: Ground-truth accuracy breakdown (`CORRECT`, `INCORRECT`, `NEUTRAL`).
