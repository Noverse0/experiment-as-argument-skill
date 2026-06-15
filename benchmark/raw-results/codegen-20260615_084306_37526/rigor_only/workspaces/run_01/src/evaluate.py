"""Cross-validation evaluation and comparison logic."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate


def cv_evaluate(X, y, model, n_splits: int = 5) -> dict:
    """Evaluate a model with TimeSeriesSplit; return per-metric mean/std/scores."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    scoring = ["roc_auc", "f1", "average_precision"]
    raw = cross_validate(model, X, y, cv=cv, scoring=scoring)

    result = {}
    for metric in scoring:
        scores = raw[f"test_{metric}"]
        result[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": [float(s) for s in scores],
            "n_folds": int(len(scores)),
        }
    return result


def compare(lr_results: dict, gbm_results: dict, primary: str = "roc_auc") -> str:
    """Declare a winner only when the gap exceeds the noise floor.

    'No detectable difference' when |gap| < max(std_lr, std_gbm).
    This is conservative but honest for n=5 folds.
    """
    lr = lr_results[primary]
    gbm = gbm_results[primary]

    gap = gbm["mean"] - lr["mean"]
    noise = max(lr["std"], gbm["std"])

    if abs(gap) < noise:
        return "no_detectable_difference"
    return "gradient_boosting" if gap > 0 else "logistic_regression"
