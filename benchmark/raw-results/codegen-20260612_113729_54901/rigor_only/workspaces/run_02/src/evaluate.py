import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

SEEDS = [0, 1, 2]
N_SPLITS = 5


def evaluate_model(model, X, y, n_splits=N_SPLITS, seeds=SEEDS):
    """Repeated stratified k-fold cross-validation.

    Scaler is fitted on each training fold and applied to its test fold —
    no information from test rows leaks into preprocessing.
    """
    records = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            m = clone(model)
            m.fit(X_train_s, y_train)

            y_proba = m.predict_proba(X_test_s)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            records.append({
                "seed": seed,
                "fold": fold_idx,
                "roc_auc": roc_auc_score(y_test, y_proba),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "avg_precision": average_precision_score(y_test, y_proba),
            })
    return records


def summarize(records):
    """Return mean and std across all folds for each metric."""
    df = pd.DataFrame(records)
    metrics = ["roc_auc", "f1", "avg_precision"]
    summary = {}
    for m in metrics:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"] = float(df[m].std())
    summary["n_folds"] = len(records)
    return summary
