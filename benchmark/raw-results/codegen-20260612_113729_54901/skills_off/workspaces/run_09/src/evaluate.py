"""Metric computation helpers."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics from true labels and predicted probabilities."""
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def summarise(fold_metrics: list) -> Dict[str, Dict[str, float]]:
    """Aggregate per-fold metric dicts into mean ± std."""
    keys = fold_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        result[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "n": len(vals),
        }
    return result
