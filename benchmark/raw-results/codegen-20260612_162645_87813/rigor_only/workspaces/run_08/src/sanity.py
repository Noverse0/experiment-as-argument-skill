"""Cheap sanity checks that must pass before any result is believed.

Each returns a dict with the measured number and a boolean ``passed`` so the
runner can record them and a test can assert on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import clean, features_and_target
from .experiment import make_models

_SHUFFLE_SEED = 12345


def _holdout(df: pd.DataFrame, include_leak: bool, seed: int):
    X, y = features_and_target(df, include_leak=include_leak)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def leakage_demo(df_raw: pd.DataFrame, seed: int = 0) -> dict:
    """Including account_status should make the task ~trivially solvable (AUC~1).
    This proves WHY we dropped it. AUC must exceed 0.99 to count as a demonstrated
    leak."""
    df = clean(df_raw)
    Xtr, Xte, ytr, yte = _holdout(df, include_leak=True, seed=seed)
    pipe = make_models(seed)["logistic_regression"]
    pipe.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {"check": "leakage_ceiling", "auc_with_leak": auc, "passed": auc > 0.99}


def clean_not_near_perfect(df_raw: pd.DataFrame, seed: int = 0) -> dict:
    """With the leak removed, a noisy task must NOT be near-perfect. AUC must be
    below 0.95, else suspect a remaining leak."""
    df = clean(df_raw)
    Xtr, Xte, ytr, yte = _holdout(df, include_leak=False, seed=seed)
    pipe = make_models(seed)["gradient_boosting"]
    pipe.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {"check": "clean_not_near_perfect", "auc_clean": auc, "passed": auc < 0.95}


def label_shuffle(df_raw: pd.DataFrame, seed: int = 0) -> dict:
    """Shuffle the labels -> the model can learn nothing -> AUC ~ 0.5.
    If it stays high, information is leaking around the labels."""
    df = clean(df_raw)
    X, y = features_and_target(df, include_leak=False)
    rng = np.random.default_rng(_SHUFFLE_SEED)
    y_shuf = pd.Series(rng.permutation(y.values), index=y.index)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_shuf, test_size=0.25, stratify=y_shuf, random_state=seed
    )
    pipe = make_models(seed)["gradient_boosting"]
    pipe.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {"check": "label_shuffle", "auc_shuffled": auc, "passed": 0.42 <= auc <= 0.58}


def beats_baseline(df_raw: pd.DataFrame, seed: int = 0) -> dict:
    """Both real models must beat a coin flip (AUC > 0.55) on clean features."""
    df = clean(df_raw)
    Xtr, Xte, ytr, yte = _holdout(df, include_leak=False, seed=seed)
    aucs = {}
    for name, pipe in make_models(seed).items():
        pipe.fit(Xtr, ytr)
        aucs[name] = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {
        "check": "beats_baseline",
        "aucs": aucs,
        "passed": all(a > 0.55 for a in aucs.values()),
    }


def run_all(df_raw: pd.DataFrame) -> list[dict]:
    return [
        leakage_demo(df_raw),
        clean_not_near_perfect(df_raw),
        label_shuffle(df_raw),
        beats_baseline(df_raw),
    ]
