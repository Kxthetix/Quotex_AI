# Outcome Tracking & Multidimensional Analysis

## 1. Decoupling Prediction Outcome from Trade Outcome

> [!NOTE]
> A critical principle of the system is the absolute separation between **Prediction Outcome** and **Trade Outcome**.

A prediction is evaluated regardless of whether:
1. The signal was vetoed by the Risk Engine.
2. The trade order failed execution limits.
3. The portfolio was already at maximum exposure.
4. Position size was modified.

This ensures uncorrupted evaluation of the predictive intelligence models, isolated from execution or risk management behavior.

## 2. Excursion Metrics (MFE / MAE)

For every evaluated prediction over horizon $H$:

- **Realized Return**:
  $$R = \frac{P_{\text{eval}} - P_{\text{entry}}}{P_{\text{entry}}}$$

- **Maximum Favorable Excursion (MFE)**:
  Measures the maximum potential profit reached during the holding horizon:
  $$\text{MFE}_{\text{BUY}} = \frac{\max(P_{\text{high}}) - P_{\text{entry}}}{P_{\text{entry}}}$$
  $$\text{MFE}_{\text{SELL}} = \frac{P_{\text{entry}} - \min(P_{\text{low}})}{P_{\text{entry}}}$$

- **Maximum Adverse Excursion (MAE)**:
  Measures the maximum drawdown or adverse movement experienced during the horizon:
  $$\text{MAE}_{\text{BUY}} = \frac{P_{\text{entry}} - \min(P_{\text{low}})}{P_{\text{entry}}}$$
  $$\text{MAE}_{\text{SELL}} = \frac{\max(P_{\text{high}}) - P_{\text{entry}}}{P_{\text{entry}}}$$

- **Prediction Error**:
  $$\text{Error} = |R_{\text{realized}} - R_{\text{expected}}|$$

## 3. Multidimensional Quality Breakdown (Section 24)

The system tracks 5 independent dimensions:

1. **Prediction Quality**: Did the market move as forecast? (Accuracy, forward returns, MFE, MAE, forecast error).
2. **Signal Quality**: Was the categorical action useful? (Conviction calibration, NO-TRADE efficiency).
3. **Risk Quality**: Did the Risk Engine preserve capital? (Veto rate, drawdown avoidance, breaker activations).
4. **Execution Quality**: Did simulated execution reflect realistic frictions? (Average slippage, spread, fees).
5. **Portfolio Result**: Did simulated equity grow? (Net P&L, Sharpe, win rate, maximum drawdown).
