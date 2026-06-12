"""Cheap sanity checks that run before we believe any comparison. Each returns a
(name, passed, detail) record so the runner can log them and fail loudly.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from .models import build_models


def baseline_floor(X_dev, y_dev, X_test, y_test) -> dict:
    """A no-information baseline must score ~0.5 AUC. Anything we build must beat it."""
    dummy = DummyClassifier(strategy="prior").fit(X_dev, y_dev)
    proba = dummy.predict_proba(X_test)[:, 1]
    # roc_auc of a constant predictor is 0.5 by definition.
    auc = float(roc_auc_score(y_test, proba)) if proba.std() > 0 else 0.5
    return {"name": "baseline_floor", "passed": abs(auc - 0.5) < 0.05, "detail": {"auc": auc}}


def label_shuffle(X_dev, y_dev, seed: int) -> dict:
    """With shuffled labels, AUC must collapse to ~0.5. If it stays high, signal
    is leaking around the labels (a leaked feature)."""
    rng = np.random.default_rng(seed)
    y_shuf = y_dev.to_numpy().copy()
    rng.shuffle(y_shuf)
    n = len(X_dev)
    cut = int(n * 0.7)
    model = clone(build_models(seed)["gradient_boosting"])
    model.fit(X_dev.iloc[:cut], y_shuf[:cut])
    proba = model.predict_proba(X_dev.iloc[cut:])[:, 1]
    auc = float(roc_auc_score(y_shuf[cut:], proba))
    return {"name": "label_shuffle", "passed": abs(auc - 0.5) < 0.1, "detail": {"auc": auc}}


def overfit_tiny_subset(X_dev, y_dev, seed: int) -> dict:
    """The model must memorize a tiny slice (train AUC ~1.0). If it cannot, the
    pipeline is wired wrong."""
    n = 60
    Xs, ys = X_dev.iloc[:n], y_dev.iloc[:n]
    model = clone(build_models(seed)["gradient_boosting"]).fit(Xs, ys)
    proba = model.predict_proba(Xs)[:, 1]
    auc = float(roc_auc_score(ys, proba))
    return {"name": "overfit_tiny_subset", "passed": auc > 0.98, "detail": {"train_auc": auc}}


def run_all(split, seed: int) -> list[dict]:
    return [
        baseline_floor(split.X_dev, split.y_dev, split.X_test, split.y_test),
        label_shuffle(split.X_dev, split.y_dev, seed),
        overfit_tiny_subset(split.X_dev, split.y_dev, seed),
    ]
