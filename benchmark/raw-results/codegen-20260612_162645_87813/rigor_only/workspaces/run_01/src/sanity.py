"""Cheap checks that run BEFORE we believe any comparison. Each returns a dict with the
measured number and a pass/fail verdict so the report can cite them instead of vibes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from . import data as D
from .models import make_gboost


def _time_holdout(df: pd.DataFrame):
    """Last-25%-by-time holdout, for quick single-split sanity checks."""
    cut = int(len(df) * 0.75)
    return df.iloc[:cut], df.iloc[cut:]


def baseline_floor(df: pd.DataFrame, seed: int) -> dict:
    """A no-skill DummyClassifier must score ~0.5 AUC. Establishes the floor real models beat."""
    train, test = _time_holdout(df)
    Xtr, ytr = D.features_target(train)
    Xte, yte = D.features_target(test)
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    proba = dummy.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    return {"check": "baseline_floor", "auc": auc, "expected": "~0.5",
            "passed": 0.45 <= auc <= 0.55}


def leakage_ceiling(df: pd.DataFrame, seed: int) -> dict:
    """AUDIT: if we (wrongly) include account_status, AUC should rocket to ~1.0 — proving it is
    a target leak and justifying its removal from the real experiment."""
    train, test = _time_holdout(df)
    enc = {"active": 0, "closed": 1}
    Xtr = train[D.FEATURES].assign(account_status=train["account_status"].map(enc))
    Xte = test[D.FEATURES].assign(account_status=test["account_status"].map(enc))
    ytr, yte = train[D.TARGET], test[D.TARGET]
    model = make_gboost(seed).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    # "passed" = the leak is real and large, confirming our decision to drop it.
    return {"check": "leakage_ceiling_audit", "auc_with_leak": auc, "expected": "~1.0",
            "passed": auc > 0.99}


def label_shuffle(df: pd.DataFrame, seed: int) -> dict:
    """Shuffle y; with honest features AUC must collapse to ~0.5. If it doesn't, information
    is leaking around the labels (e.g. an id that tracks the target)."""
    train, test = _time_holdout(df)
    Xtr, ytr = D.features_target(train)
    Xte, yte = D.features_target(test)
    rng = np.random.default_rng(seed)
    ytr_shuf = pd.Series(rng.permutation(ytr.to_numpy()), index=ytr.index)
    model = make_gboost(seed).fit(Xtr, ytr_shuf)
    auc = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    return {"check": "label_shuffle", "auc": auc, "expected": "~0.5",
            "passed": 0.40 <= auc <= 0.60}


def overfit_tiny(df: pd.DataFrame, seed: int) -> dict:
    """A capable model must (near-)memorize a tiny slice -> train AUC ~1.0. If it can't, the
    pipeline (features/labels) is broken before we ever compare arms."""
    tiny = df.head(60)
    X, y = D.features_target(tiny)
    model = make_gboost(seed).fit(X, y)
    auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    return {"check": "overfit_tiny_subset", "train_auc": auc, "expected": "~1.0",
            "passed": auc > 0.95}


def run_all(df: pd.DataFrame, seed: int) -> list[dict]:
    return [
        baseline_floor(df, seed),
        leakage_ceiling(df, seed),
        label_shuffle(df, seed),
        overfit_tiny(df, seed),
    ]
