# Model Evaluation & Calibration Specification

## 1. Overview

Phase 4 introduces standardized evaluation metrics, probability calibration, and reproducible MLflow experiment tracking.

## 2. Standardized Metrics

### 2.1 Classification Metrics (`evaluate_classifier`)
- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: Favorable positive predictive value $\frac{TP}{TP + FP}$.
- **Recall**: True positive sensitivity $\frac{TP}{TP + FN}$.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **Log Loss (Cross-Entropy)**: Uncertainty-weighted log loss of predicted probabilities.
- **Brier Score**: Mean squared error of probabilistic predictions:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

### 2.2 Regression Metrics (`evaluate_regressor`)
- **MAE**: Mean Absolute Error $\frac{1}{N} \sum |y_i - \hat{y}_i|$.
- **RMSE**: Root Mean Squared Error $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$.
- **Median Absolute Error**: Robust outlier-resistant error measure.
- **$R^2$ Score**: Coefficient of determination.
- **Correlation**: Pearson correlation coefficient between predicted and true values.

## 3. Probability Calibration (`ProbabilityCalibrator`)

Raw model scores and uncalibrated probabilities frequently exhibit overconfidence. The system fits post-hoc calibration curves **exclusively on validation predictions**:

- **Platt Scaling (`method="sigmoid"` or `"platt"`)**: Fits a logistic regression mapping uncalibrated decision probabilities to observed empirical frequencies.
- **Isotonic Regression (`method="isotonic"`)**: Fits a non-parametric monotonic piecewise-constant function.
- **Expected Calibration Error (ECE)**: Partitions predictions into $M$ equal-width probability bins and computes:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

## 4. MLflow Experiment Tracking (`ExperimentTracker`)

All training and evaluation runs are logged to MLflow (`data/mlruns` or remote tracking server):
- Parameters: `algorithm`, `feature_version`, `label_version`, `dataset_id`, hyperparameters.
- Tags: `model_name`, `model_version`, `status`, chronological `training_start` and `training_end`.
- Metrics: All classification and regression scores.
- Artifacts: Serialized `.joblib` model artifact and `.json` metadata ledger.
