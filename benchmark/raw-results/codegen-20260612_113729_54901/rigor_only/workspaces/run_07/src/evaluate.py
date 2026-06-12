"""Evaluation metrics and sanity checks."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred_proba, y_pred=None) -> dict:
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def baseline_auc(X_train, y_train, X_test, y_test) -> float:
    """Majority-class baseline: AUC should be ~0.5."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    proba = dummy.predict_proba(X_test)[:, 1]
    try:
        return float(roc_auc_score(y_test, proba))
    except ValueError:
        return 0.5


def sanity_overfit_check(pipeline, X_train, y_train, threshold: float = 0.65) -> bool:
    """Model must reach >threshold train AUC (prove it can fit the data)."""
    proba = pipeline.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, proba)
    ok = train_auc > threshold
    print(f"  [sanity] train AUC={train_auc:.4f} (overfit check {'PASS' if ok else 'FAIL'})")
    return ok


def aggregate_runs(run_metrics: list[dict]) -> dict:
    """Given a list of metric dicts, return mean ± std for each metric."""
    result = {}
    for key in run_metrics[0]:
        vals = np.array([m[key] for m in run_metrics])
        result[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": len(vals)}
    return result
