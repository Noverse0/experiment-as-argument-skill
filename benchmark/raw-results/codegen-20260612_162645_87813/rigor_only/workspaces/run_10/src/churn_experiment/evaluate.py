"""Evaluation: temporal cross-validation for the comparative claim (with
variance), plus a single final touch of the held-out test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .models import build_models


def _score(estimator, X, y) -> dict[str, float]:
    proba = estimator.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


def cross_validate_arms(
    X_dev: pd.DataFrame, y_dev: pd.Series, seed: int, n_splits: int = 5
) -> dict:
    """Run TimeSeriesSplit CV on the dev set. Each arm sees identical folds, so
    fold-paired differences are meaningful. Returns per-arm fold scores plus the
    paired GBM-minus-LogReg difference for the primary metric (roc_auc).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = list(splitter.split(X_dev))

    per_arm: dict[str, dict[str, list[float]]] = {
        name: {"roc_auc": [], "pr_auc": []} for name in build_models(seed)
    }

    for tr_idx, va_idx in folds:
        X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
        y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]
        for name, model in build_models(seed).items():
            est = clone(model).fit(X_tr, y_tr)
            s = _score(est, X_va, y_va)
            per_arm[name]["roc_auc"].append(s["roc_auc"])
            per_arm[name]["pr_auc"].append(s["pr_auc"])

    summary = {}
    for name, scores in per_arm.items():
        summary[name] = {
            metric: {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "folds": [float(v) for v in vals],
            }
            for metric, vals in scores.items()
        }

    gbm = np.array(per_arm["gradient_boosting"]["roc_auc"])
    lr = np.array(per_arm["logreg"]["roc_auc"])
    diff = gbm - lr
    paired = {
        "metric": "roc_auc",
        "mean_diff_gbm_minus_lr": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "n_folds": int(len(diff)),
        "per_fold_diff": [float(d) for d in diff],
    }
    return {"n_splits": n_splits, "per_arm": summary, "paired_roc_auc": paired}


def final_test_evaluation(split, seed: int) -> dict:
    """Refit each arm on the full dev set and score ONCE on the held-out test set.
    This is the only place the test set is touched.
    """
    out = {}
    for name, model in build_models(seed).items():
        est = clone(model).fit(split.X_dev, split.y_dev)
        out[name] = _score(est, split.X_test, split.y_test)
    return out
