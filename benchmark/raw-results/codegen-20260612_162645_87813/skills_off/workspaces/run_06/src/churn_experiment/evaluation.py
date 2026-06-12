"""Evaluation: cross-validated comparison, sanity checks, and final holdout.

Statistics are computed without scipy to keep the dependency surface minimal.
We use a paired comparison across the shared CV folds and a normal-approx 95%
confidence interval on the mean fold-wise difference.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from . import config
from .models import build_models


@dataclass
class FoldScores:
    model: str
    roc_auc_mean: float
    roc_auc_sd: float
    average_precision_mean: float
    average_precision_sd: float
    n_folds: int
    roc_auc_per_fold: list[float]


def _cv():
    return TimeSeriesSplit(n_splits=config.N_CV_SPLITS)


def cross_validated_scores(X, y, seed: int = config.SEED) -> dict[str, FoldScores]:
    """Forward-chaining CV on the (time-ordered) training set.

    X and y must already be sorted by time so TimeSeriesSplit respects it.
    Returns per-model mean +/- sd across folds for each scoring metric.
    """
    out: dict[str, FoldScores] = {}
    for name, model in build_models(seed).items():
        res = cross_validate(
            model, X, y, cv=_cv(), scoring=config.SCORING, n_jobs=1
        )
        out[name] = FoldScores(
            model=name,
            roc_auc_mean=float(np.mean(res["test_roc_auc"])),
            roc_auc_sd=float(np.std(res["test_roc_auc"], ddof=1)),
            average_precision_mean=float(np.mean(res["test_average_precision"])),
            average_precision_sd=float(np.std(res["test_average_precision"], ddof=1)),
            n_folds=config.N_CV_SPLITS,
            roc_auc_per_fold=[float(v) for v in res["test_roc_auc"]],
        )
    return out


def paired_difference(scores: dict[str, FoldScores]) -> dict:
    """Paired GBM - LR difference on the primary metric across shared folds.

    Returns mean difference, sd, a normal-approx 95% CI, and the honest verdict
    (a winner is only claimed if the CI excludes zero).
    """
    gbm = np.array(scores["gradient_boosting"].roc_auc_per_fold)
    lr = np.array(scores["logistic_regression"].roc_auc_per_fold)
    diff = gbm - lr
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    if not excludes_zero:
        verdict = "no detectable difference"
    elif mean > 0:
        verdict = "gradient_boosting better"
    else:
        verdict = "logistic_regression better"
    return {
        "metric": config.PRIMARY_METRIC,
        "comparison": "gradient_boosting - logistic_regression",
        "mean_diff": mean,
        "sd_diff": sd,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "n_folds": n,
        "verdict": verdict,
    }


# --- Sanity checks ---------------------------------------------------------

def baseline_floor(X, y, seed: int = config.SEED) -> dict:
    """Trivial baseline. ROC-AUC of a prior-only classifier must be ~0.5;
    any real model has to beat this floor to be worth anything."""
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    res = cross_validate(dummy, X, y, cv=_cv(), scoring=["roc_auc"], n_jobs=1)
    return {
        "strategy": "prior",
        "roc_auc_mean": float(np.mean(res["test_roc_auc"])),
        "majority_class_rate": float(y.value_counts(normalize=True).max()),
    }


def label_shuffle_test(X, y, seed: int = config.SEED) -> dict:
    """With labels shuffled, signal must vanish: ROC-AUC -> ~0.5. If it stays
    high, information is leaking around the labels (e.g. via the split)."""
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    model = build_models(seed)["logistic_regression"]
    res = cross_validate(model, X, y_shuf, cv=_cv(), scoring=["roc_auc"], n_jobs=1)
    return {"roc_auc_mean": float(np.mean(res["test_roc_auc"]))}


def leakage_demonstration(train_df, test_df, seed: int = config.SEED) -> dict:
    """Show WHY account_status was dropped: include it and watch ROC-AUC jump
    to ~1.0 on the holdout. This documents the trap rather than hiding it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def encode(df):
        x = df[config.FEATURES].copy()
        x["account_status_closed"] = (df["account_status"] == "closed").astype(int)
        return x

    Xtr, ytr = encode(train_df), train_df[config.TARGET].astype(int)
    Xte, yte = encode(test_df), test_df[config.TARGET].astype(int)
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))]
    ).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        "feature_set": "FEATURES + account_status (intentionally leaked)",
        "holdout_roc_auc": float(roc_auc_score(yte, proba)),
        "note": "near 1.0 confirms account_status is a target leak; excluded from the real experiment",
    }


# --- Final holdout (touched exactly once) ----------------------------------

def final_holdout(Xtr, ytr, Xte, yte, seed: int = config.SEED) -> dict[str, dict]:
    """Refit each model on the full training set and score ONCE on the
    chronologically held-out test set. No decisions are made after this."""
    out: dict[str, dict] = {}
    for name, model in build_models(seed).items():
        fitted = clone(model).fit(Xtr, ytr)
        proba = fitted.predict_proba(Xte)[:, 1]
        pred = fitted.predict(Xte)
        out[name] = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "average_precision": float(average_precision_score(yte, proba)),
            "brier_score": float(brier_score_loss(yte, proba)),
            "accuracy": float((pred == yte.to_numpy()).mean()),
        }
    return out


def fold_scores_to_dict(scores: dict[str, FoldScores]) -> dict:
    return {k: asdict(v) for k, v in scores.items()}
