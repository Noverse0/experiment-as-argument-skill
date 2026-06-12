"""Metric calculations for churn experiment."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def compute_metrics(y_true, y_pred_proba, y_pred_class) -> dict:
    """
    Compute comprehensive metrics for binary classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        y_pred_class: Predicted classes

    Returns:
        dict with metrics
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class),
        'recall': recall_score(y_true, y_pred_class),
        'accuracy': accuracy_score(y_true, y_pred_class),
    }
    return metrics
