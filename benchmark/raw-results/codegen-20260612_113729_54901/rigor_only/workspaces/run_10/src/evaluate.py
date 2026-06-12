"""Cross-validation and metric summary utilities."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.base import clone


def evaluate_model(pipeline, X, y, n_splits: int = 5) -> dict:
    """Evaluate a pipeline with temporal cross-validation.

    X and y must already be sorted by the temporal column so that each fold
    trains on earlier records and evaluates on later ones.

    Returns arrays of per-fold metrics so callers can compute mean ± std.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, f1s, accs = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(pipeline)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        accs.append(accuracy_score(y_test, y_pred))

    return {
        "roc_auc": np.array(aucs),
        "f1": np.array(f1s),
        "accuracy": np.array(accs),
    }


def summarize(metrics: dict) -> dict:
    """Convert per-fold arrays into mean/std/n/values dicts."""
    return {
        k: {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "n": int(len(v)),
            "values": [float(x) for x in v],
        }
        for k, v in metrics.items()
    }
