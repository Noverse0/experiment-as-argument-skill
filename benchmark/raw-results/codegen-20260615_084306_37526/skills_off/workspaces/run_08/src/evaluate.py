"""Cross-validation evaluation using TimeSeriesSplit."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, Any


def evaluate_pipeline(pipeline, X, y, n_splits: int = 5) -> Dict[str, Any]:
    """Evaluate a pipeline with TimeSeriesSplit CV.

    TimeSeriesSplit preserves temporal ordering: training folds always
    precede the test fold in time, matching how the model would be deployed.

    Returns per-metric dicts with mean, std, and n (number of folds).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {"roc_auc": [], "f1": [], "precision": [], "recall": []}

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))

    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for k, v in fold_metrics.items()
    }
