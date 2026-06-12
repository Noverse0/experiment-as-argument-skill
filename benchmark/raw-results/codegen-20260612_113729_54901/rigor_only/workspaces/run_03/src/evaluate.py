import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    brier_score_loss,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
    }


def aggregate(metrics_list: list[dict]) -> dict:
    """Summarise a list of per-run metric dicts into mean ± std."""
    keys = metrics_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metrics_list])),
            "std": float(np.std([m[k] for m in metrics_list])),
            "values": [m[k] for m in metrics_list],
        }
        for k in keys
    }
