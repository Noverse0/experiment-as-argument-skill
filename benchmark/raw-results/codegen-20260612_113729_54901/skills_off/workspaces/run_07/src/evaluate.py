"""Cross-validation and hold-out test evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline


def cv_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """TimeSeriesSplit CV; returns per-metric mean, std, and raw fold scores."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }
    raw = cross_validate(pipeline, X, y, cv=tscv, scoring=scoring)
    return {
        metric: {
            "mean": float(np.mean(raw[f"test_{metric}"])),
            "std": float(np.std(raw[f"test_{metric}"])),
            "scores": raw[f"test_{metric}"].tolist(),
        }
        for metric in scoring
    }


def holdout_metrics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Fit on train, evaluate on held-out test set. Call this exactly once per model."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
