"""Cross-validation evaluation with temporal splits."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


def evaluate_pipeline(pipeline, X, y, n_splits: int = 5) -> dict:
    """Evaluate a pipeline with TimeSeriesSplit to respect temporal ordering.

    Clones the pipeline each fold so state never leaks between folds.
    Returns per-fold and aggregate metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    aucs, f1s, pr_aucs = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        y_proba = fold_pipeline.predict_proba(X_test)[:, 1]
        y_pred = fold_pipeline.predict(X_test)

        aucs.append(float(roc_auc_score(y_test, y_proba)))
        f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
        pr_aucs.append(float(average_precision_score(y_test, y_proba)))

    return {
        "roc_auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "values": aucs,
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
            "values": f1s,
        },
        "pr_auc": {
            "mean": float(np.mean(pr_aucs)),
            "std": float(np.std(pr_aucs)),
            "values": pr_aucs,
        },
        "n_folds": n_splits,
    }


def label_shuffle_check(pipeline, X, y, seed: int = 42) -> float:
    """Sanity check: shuffled labels must produce near-random AUC.

    If AUC stays high after shuffling, there is label-independent signal
    leaking into the features (e.g., a leaky column was missed).
    Returns the shuffled-label ROC-AUC (should be ~0.5).
    """
    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    y_shuffled.values[:] = rng.permutation(y.values)

    n = len(X)
    split = int(n * 0.8)
    p = clone(pipeline)
    p.fit(X.iloc[:split], y_shuffled.iloc[:split])
    y_proba = p.predict_proba(X.iloc[split:])[:, 1]
    return float(roc_auc_score(y_shuffled.iloc[split:], y_proba))
