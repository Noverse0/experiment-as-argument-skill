"""Evaluate model performance with multiple metrics."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred_proba, y_pred_label=None):
    """
    Compute evaluation metrics.

    Args:
        y_true: ground truth labels
        y_pred_proba: predicted probabilities (for AUC)
        y_pred_label: predicted binary labels (optional, for F1/precision/recall)

    Returns:
        dict of metrics
    """
    if y_pred_label is None:
        y_pred_label = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)

    # Find threshold that maximizes F1 on predictions
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    precision = precision_score(y_true, y_pred_label, zero_division=0)
    recall = recall_score(y_true, y_pred_label, zero_division=0)

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def baseline_majority(y_true):
    """Predict majority class; AUC will be ~0.5 on balanced data."""
    y_pred_proba = np.full_like(y_true, y_true.mean(), dtype=float)
    return evaluate(y_true, y_pred_proba)


def baseline_random(y_true, seed=42):
    """Predict random probabilities; AUC should be near 0.5."""
    rng = np.random.default_rng(seed)
    y_pred_proba = rng.random(len(y_true))
    return evaluate(y_true, y_pred_proba)
