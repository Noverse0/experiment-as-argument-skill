"""Evaluation utilities: temporal cross-validation and metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


def _fold_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def majority_baseline_auc(y: pd.Series) -> float:
    """AUC of a majority-class predictor (constant probability = churn_rate)."""
    # A constant-probability predictor yields AUC == 0.5 exactly.
    return 0.5


def cross_validate_temporal(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """TimeSeriesSplit CV; each fold trains on past, validates on future cohort.

    Returns per-metric mean, std, and raw fold values.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {"auc": [], "f1": [], "accuracy": []}

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fresh clone-fit so folds are independent.
        from sklearn.base import clone
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_val)[:, 1]
        y_pred = pipe.predict(X_val)

        m = _fold_metrics(y_val, y_pred, y_proba)
        for k, v in m.items():
            fold_metrics[k].append(v)

    return {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)),
            "values": [float(x) for x in v],
            "n_folds": len(v),
        }
        for k, v in fold_metrics.items()
    }


def summarise(name: str, scores: dict) -> str:
    auc = scores["auc"]
    f1 = scores["f1"]
    return (
        f"{name:30s}  "
        f"AUC={auc['mean']:.4f}±{auc['std']:.4f}  "
        f"F1={f1['mean']:.4f}±{f1['std']:.4f}"
    )
