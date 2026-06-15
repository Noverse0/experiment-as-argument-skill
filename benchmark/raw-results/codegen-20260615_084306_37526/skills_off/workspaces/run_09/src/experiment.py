"""Core experiment logic: data preparation and model evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate

# Features available before the outcome is known.
# Excluded:
#   customer_id          — row identifier, no predictive signal
#   signup_date          — used for temporal ordering only; raw date not useful
#   days_since_last_login — POST-OUTCOME LEAK: a churned customer has stopped
#                           logging in by definition, so this value is recorded
#                           after the outcome. Including it is target leakage.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, parse_dates=["signup_date"])

    # Remove the 200 appended duplicate rows before any split so they cannot
    # straddle train/test.
    df = df.drop_duplicates().reset_index(drop=True)

    # Sort by signup_date to enforce temporal ordering for TimeSeriesSplit.
    df = df.sort_values("signup_date").reset_index(drop=True)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def _scoring() -> dict:
    return {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }


def cross_validate_model(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = cross_validate(
        pipeline, X, y,
        cv=tscv,
        scoring=_scoring(),
        return_train_score=False,
        n_jobs=1,
    )
    return {
        metric: {
            "mean": float(np.mean(cv[f"test_{metric}"])),
            "std": float(np.std(cv[f"test_{metric}"])),
            "values": [float(v) for v in cv[f"test_{metric}"]],
        }
        for metric in _scoring()
    }


def majority_baseline_auc(y: pd.Series, n_splits: int = 5) -> float:
    """ROC-AUC of a majority-class dummy, averaged over TimeSeriesSplit folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dummy = DummyClassifier(strategy="most_frequent")
    X_dummy = pd.DataFrame({"x": np.zeros(len(y))})
    cv = cross_validate(dummy, X_dummy, y, cv=tscv, scoring="roc_auc")
    return float(np.mean(cv["test_score"]))


def run_experiment(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    from src.pipeline import make_gb_pipeline, make_lr_pipeline

    baseline_auc = majority_baseline_auc(y, n_splits)

    results: dict = {
        "n_samples": int(len(y)),
        "churn_rate": float(y.mean()),
        "features": FEATURES,
        "n_cv_splits": n_splits,
        "baseline_roc_auc": baseline_auc,
        "models": {},
    }

    for name, pipeline in [
        ("LogisticRegression", make_lr_pipeline()),
        ("GradientBoosting", make_gb_pipeline()),
    ]:
        results["models"][name] = cross_validate_model(pipeline, X, y, n_splits)

    return results
