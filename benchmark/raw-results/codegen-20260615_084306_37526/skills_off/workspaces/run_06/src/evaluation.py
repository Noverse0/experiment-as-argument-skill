"""Cross-validation evaluation using TimeSeriesSplit.

TimeSeriesSplit ensures each fold trains on earlier signups and tests on later
signups — matching how the model would be deployed (predict churn for new cohorts
using a model trained on older ones).  StandardScaler is fitted on the train
portion of each fold only, preventing any leakage through normalization statistics.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    make_pipeline_fn: Callable,
    n_splits: int = 5,
    seeds: list[int] | None = None,
) -> dict:
    """Run TimeSeriesSplit CV over multiple seeds; return aggregated metrics."""
    if seeds is None:
        seeds = [42, 123, 7]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_aucs: list[float] = []
    all_f1s: list[float] = []

    for seed in seeds:
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Pipeline fits scaler on train only.
            pipe = make_pipeline_fn(random_state=seed)
            pipe.fit(X_train, y_train)

            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            all_aucs.append(float(roc_auc_score(y_test, proba)))
            all_f1s.append(float(f1_score(y_test, pred, zero_division=0)))

    return {
        "auc_mean": float(np.mean(all_aucs)),
        "auc_std": float(np.std(all_aucs)),
        "f1_mean": float(np.mean(all_f1s)),
        "f1_std": float(np.std(all_f1s)),
        "n_observations": len(all_aucs),
        "seeds": seeds,
        "n_splits": n_splits,
        "auc_per_run": all_aucs,
        "f1_per_run": all_f1s,
    }


def sanity_checks(X: np.ndarray, y: np.ndarray, make_pipeline_fn: Callable) -> dict:
    """Run baseline floor and label-shuffle checks.

    Label-shuffle uses an 80/20 train/test split so the shuffled-label model is
    evaluated on held-out data.  Evaluating on the same data used for training
    would show high AUC for any model that can memorise (e.g. GBM), which
    obscures real leakage.
    """
    results: dict[str, object] = {}

    n = len(y)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    results["churn_rate"] = float(y.mean())

    # Train AUC on a held-out split using real labels.
    pipe = make_pipeline_fn(random_state=0)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    results["train_auc_full"] = float(roc_auc_score(y_te, proba))

    # Label-shuffle: shuffle train labels only; evaluate on same held-out set.
    # AUC should fall toward ~0.5 because the model learns noise.
    rng = np.random.default_rng(0)
    y_shuffled_tr = rng.permutation(y_tr)
    pipe2 = make_pipeline_fn(random_state=0)
    pipe2.fit(X_tr, y_shuffled_tr)
    proba2 = pipe2.predict_proba(X_te)[:, 1]
    results["shuffled_label_auc"] = float(roc_auc_score(y_te, proba2))
    results["label_shuffle_ok"] = results["shuffled_label_auc"] < 0.65

    return results
