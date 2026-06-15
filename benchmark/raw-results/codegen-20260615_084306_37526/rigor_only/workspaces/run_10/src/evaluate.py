"""Cross-validation and sanity-check utilities."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from .pipeline import make_pipeline


def run_cv(X, y, estimator, n_splits: int = 5) -> dict:
    """Temporal cross-validation using TimeSeriesSplit.

    Data must already be sorted chronologically before calling this function.
    StandardScaler is fitted inside each fold (train only) via make_pipeline.
    """
    pipe = make_pipeline(estimator)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_validate(
        pipe, X, y, cv=tscv,
        scoring=["roc_auc", "f1"],
        return_train_score=False,
    )
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
        "n_splits": n_splits,
    }


def label_shuffle_auc(X, y, estimator, n_splits: int = 5) -> float:
    """AUC with globally permuted labels.

    Expected result: ~0.5 (random baseline). A result materially above 0.5
    indicates that information is leaking around the labels (e.g., temporal
    structure in features correlated with the sorted test set).
    """
    y_shuffled = np.random.default_rng(0).permutation(y)
    result = run_cv(X, y_shuffled, estimator, n_splits=n_splits)
    return result["roc_auc_mean"]
