"""Cross-validation evaluation using temporal (time-series) splits.

TimeSeriesSplit sorts rows by signup_date and always trains on earlier customers,
evaluating on later ones — consistent with how the model would be used in production.
Random splits would leak future information into the training set.
"""

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


def cv_evaluate(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        results.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })

    return results


def summarize(fold_results: list[dict]) -> dict:
    import numpy as np
    metrics = ["roc_auc", "pr_auc", "f1"]
    return {
        m: {
            "mean": float(np.mean([r[m] for r in fold_results])),
            "std": float(np.std([r[m] for r in fold_results])),
            "n": len(fold_results),
        }
        for m in metrics
    }
