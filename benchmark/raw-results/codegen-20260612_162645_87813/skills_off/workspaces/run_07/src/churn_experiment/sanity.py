"""Cheap sanity checks that run before believing any comparison.

Each returns a dict with a ``passed`` flag and the numbers behind it so the
report can quote measured values rather than assertions of faith.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .data import PreparedData, leakage_audit
from .evaluate import baseline_cv, time_series_cv


def check_leak_excluded(raw_df: pd.DataFrame, prepared: PreparedData) -> dict:
    """account_status must be a near-perfect leak in raw data AND absent from X."""
    audit = leakage_audit(raw_df)
    leak_frac = audit["account_status_leak_fraction"]
    excluded = "account_status" not in prepared.X.columns
    return {
        "name": "leak_excluded",
        "account_status_leak_fraction": leak_frac,
        "account_status_in_features": not excluded,
        # The planted leak is near-perfect; we require it absent from features.
        "passed": bool(excluded and leak_frac > 0.95),
    }


def check_baseline_floor(prepared: PreparedData, n_splits: int) -> dict:
    """A prior-only dummy must sit at chance (ROC-AUC ~0.5)."""
    res = baseline_cv(prepared.X, prepared.y, n_splits)
    auc = res.mean_std()["roc_auc"]["mean"]
    return {
        "name": "baseline_floor",
        "baseline_roc_auc": auc,
        "passed": bool(0.45 <= auc <= 0.55),
    }


def check_label_shuffle(model_factory, prepared: PreparedData, n_splits: int) -> dict:
    """With permuted training labels, ROC-AUC must collapse to ~0.5.

    A higher value means information is leaking around the labels.
    """
    res = time_series_cv(
        model_factory, prepared.X, prepared.y, n_splits, shuffle_labels=True, shuffle_seed=123
    )
    auc = res.mean_std()["roc_auc"]["mean"]
    return {
        "name": "label_shuffle",
        "shuffled_roc_auc": auc,
        "passed": bool(auc <= 0.58),
    }


def check_overfit_tiny(model_factory, prepared: PreparedData, n: int = 60) -> dict:
    """The model must (over)fit a tiny slice: train ROC-AUC near 1.0.

    If it cannot memorize 60 rows, the pipeline is broken.
    """
    Xs = prepared.X.iloc[:n]
    ys = prepared.y.iloc[:n].to_numpy()
    est = clone(model_factory())
    est.fit(Xs, ys)
    proba = est.predict_proba(Xs)[:, 1]
    auc = float(roc_auc_score(ys, proba))
    return {
        "name": "overfit_tiny",
        "train_roc_auc": auc,
        "n": n,
        "passed": bool(auc >= 0.90),
    }
