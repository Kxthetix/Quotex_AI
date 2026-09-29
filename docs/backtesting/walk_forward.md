# Phase 5 — Rolling Walk-Forward Validation

## 1. Methodology and Rationale

Static single-split train/test evaluations can severely overfit market regimes and fail to detect parameter decay. The Phase 5 `WalkForwardEngine` (`backtesting/walk_forward.py`) implements chronological rolling out-of-sample cross-validation:

```
TRAIN [0 : W_train]
  |
  v
VALIDATE [W_train : W_train + W_val]
  |
  v
TEST (Out-of-Sample) [W_train + W_val : W_train + W_val + W_test]
  |
  v
ROLL FORWARD by Step Size
  |
  v
REPEAT
```

## 2. Configuration Parameters

- `train_bars`: Historical bars used exclusively for fitting preprocessing and model parameters.
- `val_bars`: Validation bars used exclusively for model tuning and hyperparameter freezing.
- `test_bars`: Strict out-of-sample bars evaluated by the backtesting engine.
- `step_bars`: Stride / roll step size to advance the window.

## 3. Strict Temporal Isolation (Section 43 Invariant)

Every walk-forward window enforces:
$$\max(T_{train}) < \min(T_{val}) < \min(T_{test})$$

- Test period results NEVER alter model parameters for that test window.
- Out-of-sample trades are accumulated across windows to compute aggregate walk-forward Sharpe, Sortino, Win Rate, and Return.
- Out-of-sample equity curves are stitched sequentially across windows.
