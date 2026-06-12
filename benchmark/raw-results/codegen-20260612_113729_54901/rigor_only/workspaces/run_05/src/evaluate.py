"""Evaluation utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate per-seed metrics into mean ± std."""
    keys = runs[0].keys()
    result = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        result[f"{k}_mean"] = float(vals.mean())
        result[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        result["n"] = len(vals)
    return result
