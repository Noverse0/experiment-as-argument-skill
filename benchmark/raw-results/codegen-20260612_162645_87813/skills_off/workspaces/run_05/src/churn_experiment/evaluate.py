"""Evaluation methodology and sanity checks.

Methodology
-----------
Churn prediction is forward-looking, so we evaluate with a *time-ordered*
cross-validation (``TimeSeriesSplit``) over data sorted by ``signup_date``.
Each fold trains on earlier signups and tests on strictly later ones, which
mirrors deploying a model trained on the past to score the future and never
lets a test row influence its own training fold.

Using N folds also gives N paired measurements per arm, so we can report
mean +/- standard deviation instead of a single-split anecdote, and compare
the two arms on the *same* folds (a paired comparison).

Metrics
-------
The positive class (churn) is the minority (~27%), so accuracy alone is
misleading. We report ROC-AUC and average precision (PR-AUC), which are
threshold-free and robust to imbalance, plus accuracy and the base churn
rate for context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import PreparedData
from .pipeline import MODEL_FACTORIES

N_SPLITS = 5


@dataclass
class ArmResult:
    """Cross-validated metrics for one model arm."""

    model: str
    roc_auc_mean: float
    roc_auc_sd: float
    pr_auc_mean: float
    pr_auc_sd: float
    accuracy_mean: float
    accuracy_sd: float
    n_folds: int
    roc_auc_per_fold: list[float]


def _score_fold(pipe, X_tr, y_tr, X_te, y_te) -> dict[str, float]:
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    preds = pipe.predict(X_te)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "accuracy": accuracy_score(y_te, preds),
    }


def evaluate_arm(name: str, seed: int, data: PreparedData) -> ArmResult:
    """Run time-ordered CV for a single arm and aggregate fold metrics."""
    pipe = MODEL_FACTORIES[name](seed)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    X, y = data.X, data.y

    rows = []
    for tr_idx, te_idx in splitter.split(X):
        fold = _score_fold(
            clone(pipe),
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            X.iloc[te_idx],
            y.iloc[te_idx],
        )
        rows.append(fold)

    roc = [r["roc_auc"] for r in rows]
    pr = [r["pr_auc"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    return ArmResult(
        model=name,
        roc_auc_mean=float(np.mean(roc)),
        roc_auc_sd=float(np.std(roc, ddof=1)),
        pr_auc_mean=float(np.mean(pr)),
        pr_auc_sd=float(np.std(pr, ddof=1)),
        accuracy_mean=float(np.mean(acc)),
        accuracy_sd=float(np.std(acc, ddof=1)),
        n_folds=len(rows),
        roc_auc_per_fold=[float(x) for x in roc],
    )


def paired_difference(arm_a: ArmResult, arm_b: ArmResult) -> dict[str, float]:
    """Per-fold ROC-AUC difference (a - b) with its spread.

    Folds are paired (same train/test indices for both arms), so the per-fold
    difference is a meaningful within-fold contrast. We report the mean
    difference and its sd; if the spread overlaps zero the honest conclusion
    is "no detectable difference".
    """
    diffs = np.array(arm_a.roc_auc_per_fold) - np.array(arm_b.roc_auc_per_fold)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    return {
        "metric": "roc_auc",
        "arm_a": arm_a.model,
        "arm_b": arm_b.model,
        "mean_diff_a_minus_b": mean,
        "sd_diff": sd,
        "per_fold_diff": [float(x) for x in diffs],
        # crude paired contrast: does the +/-1sd band clear zero?
        "band_excludes_zero": bool(abs(mean) > sd),
    }


# --------------------------------------------------------------------------
# Sanity checks. These do not prove the result; they catch silent pipeline
# bugs and leakage before we believe any comparison.
# --------------------------------------------------------------------------


def baseline_floor(data: PreparedData) -> dict[str, float]:
    """Trivial baseline. A 'most frequent' classifier should sit at ~0.5 AUC.

    Any real model must beat this floor to be worth anything.
    """
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for tr_idx, te_idx in splitter.split(data.X):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(data.X.iloc[tr_idx], data.y.iloc[tr_idx])
        proba = dummy.predict_proba(data.X.iloc[te_idx])[:, 1]
        # prior strategy yields constant proba -> AUC is exactly 0.5
        aucs.append(roc_auc_score(data.y.iloc[te_idx], proba))
    return {"strategy": "prior", "roc_auc_mean": float(np.mean(aucs))}


def label_shuffle_auc(name: str, seed: int, data: PreparedData) -> float:
    """Train on shuffled labels; AUC must collapse to ~0.5.

    If a model still scores well with the labels destroyed, information is
    leaking around the target and the comparison cannot be trusted.
    """
    rng = np.random.default_rng(seed)
    y_shuf = data.y.to_numpy().copy()
    rng.shuffle(y_shuf)
    shuffled = PreparedData(
        X=data.X,
        y=pd.Series(y_shuf, index=data.y.index),
        time=data.time,
        n_raw=data.n_raw,
        n_duplicates_dropped=data.n_duplicates_dropped,
    )
    return evaluate_arm(name, seed, shuffled).roc_auc_mean


def overfit_tiny_subset(name: str, seed: int, data: PreparedData, n: int = 40) -> float:
    """The model must be able to (nearly) memorise a tiny slice.

    If it cannot reach high train AUC on a handful of rows, the pipeline is
    broken (mis-wired features, constant inputs, etc.).
    """
    pipe = MODEL_FACTORIES[name](seed)
    X_small = data.X.iloc[:n]
    y_small = data.y.iloc[:n]
    pipe.fit(X_small, y_small)
    proba = pipe.predict_proba(X_small)[:, 1]
    return float(roc_auc_score(y_small, proba))


def arm_result_to_dict(r: ArmResult) -> dict:
    return asdict(r)
