"""Evaluation: time-aware cross-validation, sanity checks, and arm comparison.

The evaluation methodology is chosen to make the comparison an *argument*, not a
single lucky number:

  * Time-based CV (sklearn TimeSeriesSplit) on signup-date-ordered data. Each fold
    trains on the past and tests on the strictly-later future, matching the
    forward-looking churn task. This also yields n=N_SPLITS measurements per arm,
    so we can report mean +/- sd instead of one anecdote.
  * Threshold-free, imbalance-robust primary metric: ROC-AUC. We also report
    average precision (PR-AUC) because the positive class is the minority (~27%),
    plus the no-skill DummyClassifier baseline as a floor.
  * The two arms are compared on the SAME folds, so a paired test is appropriate.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .pipeline import SEED, make_pipeline

N_SPLITS = 5
PRIMARY_METRIC = "roc_auc"


def _score_fold(estimator, X_tr, y_tr, X_te, y_te) -> dict[str, float]:
    """Fit on a single train fold, score on its held-out future fold."""
    est = clone(estimator)
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "average_precision": float(average_precision_score(y_te, proba)),
    }


def cross_validate_arm(arm: str, X, y, n_splits: int = N_SPLITS) -> dict:
    """Run time-aware CV for one arm and return per-fold and aggregate metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipe = make_pipeline(arm)

    per_fold: list[dict[str, float]] = []
    for train_idx, test_idx in tscv.split(X):
        scores = _score_fold(
            pipe,
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[test_idx], y.iloc[test_idx],
        )
        per_fold.append(scores)

    agg = {}
    for metric in ("roc_auc", "average_precision"):
        vals = np.array([f[metric] for f in per_fold], dtype=float)
        agg[metric] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "values": vals.tolist(),
        }
    return {"arm": arm, "n_folds": n_splits, "per_fold": per_fold, "aggregate": agg}


def baseline_floor(X, y, n_splits: int = N_SPLITS) -> dict:
    """No-skill baseline (predicts the majority/prior). ROC-AUC must be ~0.5."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = dummy.predict_proba(X.iloc[test_idx])[:, 1]
        # A constant predictor is undefined for AUC ranking; guard it.
        try:
            aucs.append(float(roc_auc_score(y.iloc[test_idx], proba)))
        except ValueError:
            aucs.append(0.5)
    return {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_values": aucs}


def compare_arms(arm_a: dict, arm_b: dict, metric: str = PRIMARY_METRIC) -> dict:
    """Paired comparison of two arms on the SAME folds.

    Returns the signed difference (a - b) with its uncertainty and a paired
    t-test. The interpretation deliberately refuses a winner claim when the
    spread overlaps zero.
    """
    a = np.array(arm_a["aggregate"][metric]["values"], dtype=float)
    b = np.array(arm_b["aggregate"][metric]["values"], dtype=float)
    diff = a - b
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    n = len(diff)

    # Paired t-test across folds. With few folds this is low-powered; we report it
    # as evidence strength, not as a gatekeeper.
    if np.allclose(diff, 0.0):
        t_stat, p_value = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(a, b)
        t_stat, p_value = float(t_res.statistic), float(t_res.pvalue)

    se = sd_diff / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and se > 0:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci95 = [mean_diff - tcrit * se, mean_diff + tcrit * se]
    else:
        ci95 = [float("nan"), float("nan")]

    significant = bool(p_value < 0.05) and not np.isnan(p_value)
    if not significant:
        conclusion = "no detectable difference"
    elif mean_diff > 0:
        conclusion = f"{arm_a['arm']} outperforms {arm_b['arm']}"
    else:
        conclusion = f"{arm_b['arm']} outperforms {arm_a['arm']}"

    return {
        "metric": metric,
        "arm_a": arm_a["arm"],
        "arm_b": arm_b["arm"],
        "mean_diff_a_minus_b": mean_diff,
        "sd_diff": sd_diff,
        "ci95_diff": ci95,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_folds": n,
        "significant_at_0.05": significant,
        "conclusion": conclusion,
    }


# --------------------------------------------------------------------------- #
# Sanity checks. These cost seconds and catch silent pipeline bugs / leakage.
# --------------------------------------------------------------------------- #

def leakage_ceiling_auc(path: str) -> float:
    """Refit logreg WITH the leaked account_status column to show the ceiling.

    If account_status leaks the label, AUC should be ~1.0. This is *why* the
    column is dropped in the real experiment; we measure it rather than assert it.
    """
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression as LR

    from .data import FEATURES, TARGET, TIME_COL

    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    X = df[FEATURES + ["account_status"]]
    y = df[TARGET].astype(int)
    n = len(df)
    cut = int(n * 0.8)

    pre = ColumnTransformer(
        [("status", OneHotEncoder(handle_unknown="ignore"), ["account_status"])],
        remainder="passthrough",
    )
    pipe = SkPipeline([("pre", pre), ("m", LR(max_iter=1000, random_state=SEED))])
    pipe.fit(X.iloc[:cut], y.iloc[:cut])
    proba = pipe.predict_proba(X.iloc[cut:])[:, 1]
    return float(roc_auc_score(y.iloc[cut:], proba))


def label_shuffle_auc(arm: str, X, y, seed: int = SEED) -> float:
    """Train on shuffled labels; mean CV AUC must collapse toward 0.5.

    A value materially above 0.5 means information is leaking around the labels.
    """
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    y_shuf[:] = rng.permutation(y.values)
    res = cross_validate_arm(arm, X, y_shuf)
    return float(res["aggregate"]["roc_auc"]["mean"])


def overfit_tiny_subset_auc(arm: str, X, y, k: int = 40) -> float:
    """Model must (near-)memorise a tiny slice: train AUC on k rows should be ~1.0.

    If it cannot fit a handful of points, the pipeline itself is broken.
    """
    Xs, ys = X.iloc[:k], y.iloc[:k]
    # Need both classes present to score AUC; expand until we have them.
    while ys.nunique() < 2 and k < len(X):
        k += 20
        Xs, ys = X.iloc[:k], y.iloc[:k]
    pipe = make_pipeline(arm)
    pipe.fit(Xs, ys)
    proba = pipe.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(ys, proba))
