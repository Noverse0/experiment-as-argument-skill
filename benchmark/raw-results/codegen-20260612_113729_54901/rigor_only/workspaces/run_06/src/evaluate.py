"""Evaluation utilities: metrics, cross-validation runner, summary statistics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics for a single fold/run."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def run_cv(pipeline_fn, X: np.ndarray, y: np.ndarray, cv, random_state: int = 0) -> list:
    """
    Evaluate a model over cross-validation folds.

    Args:
        pipeline_fn: callable(random_state) -> fitted sklearn Pipeline
        X, y: full dataset (sorted by time for TimeSeriesSplit)
        cv: sklearn CV splitter (e.g. TimeSeriesSplit)
        random_state: fixed seed for model initialization

    Returns:
        List of per-fold metric dicts.
    """
    results = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe = pipeline_fn(random_state=random_state)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results.append(compute_metrics(y_test, y_pred, y_prob))
    return results


def summarize_runs(runs: list) -> dict:
    """Compute mean ± sd for each metric across runs/folds."""
    keys = list(runs[0].keys())
    summary = {}
    for k in keys:
        vals = [r[k] for r in runs]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_sd"] = float(np.std(vals, ddof=0))
        summary[f"{k}_n"] = len(vals)
    return summary
