"""Cross-validation harness for the churn experiment."""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def run_cv(
    pipeline_fn: Callable[..., Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Evaluate a pipeline with RepeatedStratifiedKFold.

    The scaler lives inside the pipeline, so it is fitted on each fold's
    training split only — no test-set leakage from preprocessing.

    Returns a dict keyed by metric name, each containing mean/std/n/all values.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}

    raw = cross_validate(
        pipeline_fn(random_state=random_state),
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    summary = {}
    for metric in scoring:
        scores = raw[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": int(len(scores)),
            "all": [float(s) for s in scores],
        }
    return summary
