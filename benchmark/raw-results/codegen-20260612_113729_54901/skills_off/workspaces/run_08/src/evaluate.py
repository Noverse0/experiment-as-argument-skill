"""Evaluation utilities: temporal CV and sanity checks."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score


def cv_evaluate(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Temporal cross-validation. Scaler is re-fit inside each fold on train only."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, f1s = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))

    return {
        "auc_per_fold": aucs,
        "f1_per_fold": f1s,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_folds": n_splits,
    }


def label_shuffle_auc(pipeline, X: pd.DataFrame, y: pd.Series,
                      n_splits: int = 5, random_state: int = 42) -> dict:
    """Shuffled-label AUC must fall to ~0.5; higher values indicate leakage."""
    rng = np.random.default_rng(random_state)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    return cv_evaluate(pipeline, X, y_shuffled, n_splits=n_splits)
