"""Cheap sanity checks that run before believing the comparison.

Each returns a small dict so the entrypoint can record outcomes in an artifact.
These catch the silent bugs that make an experiment "run but prove nothing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import FEATURES, TARGET, load_churn
from .pipeline import make_pipeline


def baseline_floor(data) -> dict:
    """A constant predictor scores AUC 0.5; AP equals prevalence. Models beat it."""
    return {"roc_auc": 0.5, "avg_precision": float(data.y.mean())}


def leakage_ceiling(csv_path: str, seed: int = 0) -> dict:
    """Re-introducing account_status should drive AUC to ~1.0 — proof it's a leak.

    This is why the column is dropped: a feature that yields a near-perfect score
    on a noisy task is leaking the label.
    """
    raw = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    leak = (raw["account_status"] == "closed").astype(int).to_numpy().reshape(-1, 1)
    y = raw[TARGET].astype(int)
    n = len(raw)
    cut = int(n * 0.7)
    pipe = make_pipeline("logreg", seed)
    pipe.fit(leak[:cut], y.iloc[:cut])
    proba = pipe.predict_proba(leak[cut:])[:, 1]
    return {"roc_auc_with_leak": float(roc_auc_score(y.iloc[cut:], proba))}


def label_shuffle(data, seed: int = 0) -> dict:
    """Shuffled labels must collapse performance to the ~0.5 floor.

    If a model still scores well on permuted labels, information is leaking around
    the labels (e.g. via the split). Here it should fall to chance.
    """
    rng = np.random.default_rng(seed)
    y = data.y.to_numpy().copy()
    rng.shuffle(y)
    cut = int(len(y) * 0.7)
    out = {}
    for arm in ("logreg", "gboost"):
        pipe = make_pipeline(arm, seed)
        pipe.fit(data.X.iloc[:cut], y[:cut])
        proba = pipe.predict_proba(data.X.iloc[cut:])[:, 1]
        out[arm] = float(roc_auc_score(y[cut:], proba))
    return out


def overfit_tiny(data, seed: int = 0, n: int = 60) -> dict:
    """On a tiny slice the models must fit train near-perfectly (pipeline works)."""
    Xs, ys = data.X.iloc[:n], data.y.iloc[:n]
    out = {}
    for arm in ("logreg", "gboost"):
        pipe = make_pipeline(arm, seed)
        pipe.fit(Xs, ys)
        proba = pipe.predict_proba(Xs)[:, 1]
        out[arm] = float(roc_auc_score(ys, proba))
    return out


def determinism(data, seed: int = 0) -> dict:
    """Same seed, same data => identical metric. Flags hidden nondeterminism."""
    cut = int(len(data.y) * 0.7)
    scores = []
    for _ in range(2):
        pipe = make_pipeline("gboost", seed)
        pipe.fit(data.X.iloc[:cut], data.y.iloc[:cut])
        proba = pipe.predict_proba(data.X.iloc[cut:])[:, 1]
        scores.append(float(roc_auc_score(data.y.iloc[cut:], proba)))
    return {"run1": scores[0], "run2": scores[1], "identical": scores[0] == scores[1]}


def run_all(csv_path: str) -> dict:
    data = load_churn(csv_path)
    return {
        "baseline_floor": baseline_floor(data),
        "leakage_ceiling": leakage_ceiling(csv_path),
        "label_shuffle": label_shuffle(data),
        "overfit_tiny": overfit_tiny(data),
        "determinism": determinism(data),
    }
