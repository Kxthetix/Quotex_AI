# Candle Strategy What-If Simulation

`scripts/run_candle_scenarios.py` compares a small set of rule-confluence thresholds for the existing closed-candle trend-pullback strategy. It adds a disciplined scenario workflow to the existing candle analyzer and payout-aware paper backtester; it does not send orders to a broker.

## Decision logic

The signal engine evaluates completed candles using trend alignment, retest of the 21-period EMA, candle close confirmation, RSI and MACD momentum, ATR distance from the mean, Bollinger Bands, and candle shape. A setup that does not meet the rule gates is `NO_TRADE`. The composite score is a measure of rule confluence, not a probability of winning.

The evaluation follows this sequence:

1. Keep data in chronological order and split it into 60% training context, 20% validation, and 20% final test.
2. Compare the chosen thresholds only in validation. Select by net expectancy per trade, with drawdown and trade count as tie-breakers.
3. If no threshold reaches the configured minimum validation trade count, mark the choice as exploratory.
4. Evaluate the selected threshold once in the final period, with no parameter selection using final-period outcomes.
5. Re-run the same final period at a payout five percentage points lower to show payout sensitivity.

For fixed-expiry binary contracts with net payout `r`, the break-even win rate is `1 / (1 + r)`. The report displays this benchmark alongside the results. A win rate above 50% alone does not establish positive expectancy.

## Run

```powershell
python scripts/run_candle_scenarios.py `
  --data-file path/to/completed_candles.parquet `
  --payout 0.85 `
  --expiry-bars 1 `
  --max-bars 1000 `
  --thresholds 0.55 0.65 0.75 `
  --min-validation-trades 20
```

`--max-bars` optionally limits the run to the most recent N candles to keep exploratory runs bounded; it must be at least 180. Use completed candles from the instrument and timeframe that match the intended contract. A crypto spot series is useful for checking software behavior, but cannot validate a separate OTC broker feed. Synthetic candles are smoke-test data only.

## Capital and validation guardrails

- Keep fixed stake small and capped; do not use martingale or loss-recovery sizing.
- Review expectancy, drawdown, profit factor, and trade count together with win rate.
- Preserve a truly later test period and avoid repeated threshold searches against it.
- Check that the winning threshold still has adequate sample size; sparse results are inconclusive.
- Treat a positive historical result as a research observation, not a return forecast or deployment approval.

The current simulation assumes next-bar-open entry, close-price settlement at the specified expiry, fixed payout, and conservative loss treatment for ties. It does not model changing broker payouts, rejected orders, broker-specific price construction, or latency within an OTC feed.
