"""Evaluation helpers for the churn experiment."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Return mean and std over repeated runs."""
    keys = runs[0].keys()
    return {
        k: {"mean": float(np.mean([r[k] for r in runs])),
            "std": float(np.std([r[k] for r in runs], ddof=0))}
        for k in keys
    }
