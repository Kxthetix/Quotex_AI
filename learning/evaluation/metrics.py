"""Standardized model evaluation metrics for Phase 4.

Computes comprehensive classification, regression, and probabilistic calibration metrics.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


@dataclass
class ClassificationMetrics:
    """Standardized classification performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    log_loss: float | None
    brier_score: float | None

    def to_dict(self) -> dict[str, float]:
        res = asdict(self)
        return {k: round(v, 6) for k, v in res.items() if v is not None}


@dataclass
class RegressionMetrics:
    """Standardized regression performance metrics."""

    mae: float
    rmse: float
    median_ae: float
    r2: float
    correlation: float

    def to_dict(self) -> dict[str, float]:
        res = asdict(self)
        return {k: round(v, 6) for k, v in res.items() if not np.isnan(v)}


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Calculate comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc_val = None
    ll_val = None
    brier_val = None

    if y_prob is not None:
        # If 2D prob array, extract class 1 probabilities
        p1 = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
        try:
            if len(np.unique(y_true)) > 1:
                auc_val = float(roc_auc_score(y_true, p1))
                ll_val = float(log_loss(y_true, y_prob))
                brier_val = float(brier_score_loss(y_true, p1))
        except Exception:
            pass

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc_val,
        log_loss=ll_val,
        brier_score=brier_val,
    )


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate comprehensive regression metrics."""
    clean_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[clean_mask]
    yp = y_pred[clean_mask]

    if len(yt) == 0:
        return RegressionMetrics(mae=0.0, rmse=0.0, median_ae=0.0, r2=0.0, correlation=0.0)

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    med_ae = float(median_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0

    # Pearson correlation
    corr = 0.0
    if len(yt) > 2 and np.std(yt) > 1e-9 and np.std(yp) > 1e-9:
        c_mat = np.corrcoef(yt, yp)
        corr = float(c_mat[0, 1])

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        median_ae=med_ae,
        r2=r2,
        correlation=corr,
    )
