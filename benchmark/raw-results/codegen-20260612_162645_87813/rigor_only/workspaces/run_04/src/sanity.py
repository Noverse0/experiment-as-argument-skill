"""Cheap sanity checks that run before believing any comparison.

Each returns a plain dict so results can be serialized and asserted in tests.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from .data import PreparedData, load_leaky_features
from .models import make_gbm


def baseline_floor(data: PreparedData, seed: int) -> float:
    """A no-information baseline must score ~0.5 AUC. Models must beat it."""
    n = len(data.y)
    cut = int(n * 0.8)
    X_tr, X_te = data.X.iloc[:cut], data.X.iloc[cut:]
    y_tr, y_te = data.y.iloc[:cut], data.y.iloc[cut:]
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(X_tr, y_tr)
    proba = dummy.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


def label_shuffle_auc(data: PreparedData, seed: int) -> float:
    """With labels shuffled, a real model must collapse to ~0.5 AUC.

    If it does not, information is leaking around the labels.
    """
    rng = np.random.default_rng(seed)
    n = len(data.y)
    cut = int(n * 0.8)
    y_shuf = data.y.to_numpy().copy()
    rng.shuffle(y_shuf)
    model = make_gbm(seed)
    model.fit(data.X.iloc[:cut], y_shuf[:cut])
    proba = model.predict_proba(data.X.iloc[cut:])[:, 1]
    return float(roc_auc_score(y_shuf[cut:], proba))


def overfit_tiny_subset(data: PreparedData, seed: int, n: int = 60) -> float:
    """The model must be able to (over)fit a tiny slice to near-perfect train
    AUC. If it cannot, the pipeline/labels are broken."""
    model = make_gbm(seed)
    Xs, ys = data.X.iloc[:n], data.y.iloc[:n]
    model.fit(Xs, ys)
    proba = model.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(ys, proba))


def leakage_ceiling(path: str, seed: int) -> float:
    """Train WITH the dropped account_status leak; AUC should be ~1.0.

    Demonstrates that the column we excluded is a genuine target leak.
    """
    X, y = load_leaky_features(path)
    n = len(y)
    cut = int(n * 0.8)
    model = make_gbm(seed)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    proba = model.predict_proba(X.iloc[cut:])[:, 1]
    return float(roc_auc_score(y.iloc[cut:], proba))


def run_all(data: PreparedData, path: str, seed: int) -> dict:
    return {
        "baseline_floor_auc": baseline_floor(data, seed),
        "label_shuffle_auc": label_shuffle_auc(data, seed),
        "overfit_tiny_train_auc": overfit_tiny_subset(data, seed),
        "leakage_ceiling_auc": leakage_ceiling(path, seed),
    }
