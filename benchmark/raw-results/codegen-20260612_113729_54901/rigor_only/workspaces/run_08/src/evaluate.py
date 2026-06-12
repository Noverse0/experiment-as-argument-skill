"""Cross-validation evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


SCORING = {
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision",
    "f1": "f1",
}


def cv_evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified k-fold cross-validation on training data.

    Returns per-metric dict with mean, std, n_folds, and per_fold scores.
    All preprocessing inside the pipeline is fit only on each fold's train portion.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, return_train_score=False
    )

    summary = {}
    for metric in ["roc_auc", "avg_precision", "f1"]:
        scores = result[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_folds": n_splits,
            "per_fold": [float(s) for s in scores],
        }
    return summary
