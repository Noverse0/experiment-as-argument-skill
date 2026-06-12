"""Evaluation: time-series cross-validation, metrics, and sanity checks.

The comparison uses a blocked time-series CV (sklearn TimeSeriesSplit). Each
fold trains on an earlier block of signups and tests on a strictly later block,
which respects the forward-looking nature of churn prediction and yields n=5
paired estimates per model so we can report mean +/- sd instead of a single
anecdotal number.

The scaler lives inside each pipeline, so it is re-fitted on the training rows
of every fold only -- preprocessing never sees the test rows (split before
transform).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

N_SPLITS = 5


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    test_churn_rate: float
    roc_auc: float
    average_precision: float


@dataclass
class ArmResult:
    name: str
    folds: list[FoldMetrics] = field(default_factory=list)

    def _vals(self, attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f in self.folds], dtype=float)

    def mean_sd(self, attr: str) -> tuple[float, float]:
        vals = self._vals(attr)
        # sample sd (ddof=1): we are estimating spread across folds.
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return float(vals.mean()), sd

    def to_dict(self) -> dict:
        roc_mean, roc_sd = self.mean_sd("roc_auc")
        ap_mean, ap_sd = self.mean_sd("average_precision")
        return {
            "name": self.name,
            "n_folds": len(self.folds),
            "roc_auc_mean": roc_mean,
            "roc_auc_sd": roc_sd,
            "roc_auc_per_fold": self._vals("roc_auc").tolist(),
            "average_precision_mean": ap_mean,
            "average_precision_sd": ap_sd,
            "average_precision_per_fold": self._vals("average_precision").tolist(),
        }


def _fold_indices(n: int, n_splits: int = N_SPLITS) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
) -> ArmResult:
    """Time-series CV for one model arm. Returns per-fold ROC-AUC and AP."""
    result = ArmResult(name=name)
    splitter = _fold_indices(len(X), n_splits)
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_arr)):
        model = clone(pipeline)
        model.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
        proba = model.predict_proba(X_arr.iloc[test_idx])[:, 1]
        y_test = y_arr.iloc[test_idx]
        result.folds.append(
            FoldMetrics(
                fold=fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
                test_churn_rate=float(y_test.mean()),
                roc_auc=float(roc_auc_score(y_test, proba)),
                average_precision=float(average_precision_score(y_test, proba)),
            )
        )
    return result


def paired_auc_per_fold(arms: dict[str, ArmResult]) -> dict:
    """Paired per-fold ROC-AUC differences between the two arms.

    Returns the per-fold differences, their mean/sd, and a paired t-test on the
    differences. With only n=5 folds this is a weak test by design -- it exists
    to keep us honest about whether any gap is distinguishable from noise, not
    to manufacture significance.
    """
    names = list(arms.keys())
    if len(names) != 2:
        raise ValueError("paired comparison expects exactly two arms")
    a, b = names
    fa = arms[a]._vals("roc_auc")
    fb = arms[b]._vals("roc_auc")
    diff = fb - fa  # positive => second arm (b) higher
    out = {
        "arm_a": a,
        "arm_b": b,
        "direction": f"positive means {b} > {a}",
        "per_fold_diff": diff.tolist(),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
    }
    # Paired t-test (scipy optional). Degrade gracefully if scipy is absent.
    try:
        from scipy import stats

        t_stat, p_val = stats.ttest_rel(fb, fa)
        out["paired_t_stat"] = float(t_stat)
        out["paired_p_value"] = float(p_val)
    except Exception:  # pragma: no cover - scipy is a declared dependency
        out["paired_t_stat"] = None
        out["paired_p_value"] = None
    return out


# --------------------------------------------------------------------------
# Sanity checks. Each returns a small dict and a boolean `passed`.
# --------------------------------------------------------------------------


def baseline_floor(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """A majority-class baseline must sit at ROC-AUC ~ 0.5. Models must beat it."""
    dummy = DummyClassifier(strategy="most_frequent")
    aucs = []
    splitter = _fold_indices(len(X), n_splits)
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    for train_idx, test_idx in splitter.split(X_arr):
        dummy.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
        proba = dummy.predict_proba(X_arr.iloc[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y_arr.iloc[test_idx], proba)))
    mean_auc = float(np.mean(aucs))
    return {
        "check": "baseline_floor",
        "mean_roc_auc": mean_auc,
        "passed": abs(mean_auc - 0.5) < 0.05,
        "note": "DummyClassifier(most_frequent) should give ROC-AUC ~ 0.5",
    }


def label_shuffle_test(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    n_splits: int = N_SPLITS,
) -> dict:
    """With labels shuffled, real predictive power must collapse to ~0.5 AUC.

    If it does not, information is leaking around the labels.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = pd.Series(rng.permutation(y.to_numpy()))
    res = evaluate_model("shuffled", pipeline, X, y_shuffled, n_splits=n_splits)
    mean_auc, _ = res.mean_sd("roc_auc")
    return {
        "check": "label_shuffle",
        "mean_roc_auc": mean_auc,
        "passed": abs(mean_auc - 0.5) < 0.07,
        "note": "Shuffled labels should destroy signal: ROC-AUC ~ 0.5",
    }


def overfit_tiny_subset(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n: int = 60
) -> dict:
    """A capable model must (near-)memorize a tiny slice -> high TRAIN AUC.

    A small slice can lack both classes; if so we widen until both appear.
    """
    idx = list(range(min(n, len(X))))
    # ensure both classes are present in the slice
    if y.iloc[idx].nunique() < 2:
        pos = y[y == 1].index[:1].tolist()
        neg = y[y == 0].index[:1].tolist()
        idx = sorted(set(idx) | set(pos) | set(neg))
    model = clone(pipeline)
    Xs = X.iloc[idx]
    ys = y.iloc[idx]
    model.fit(Xs, ys)
    proba = model.predict_proba(Xs)[:, 1]
    train_auc = float(roc_auc_score(ys, proba))
    return {
        "check": "overfit_tiny_subset",
        "n": len(idx),
        "train_roc_auc": train_auc,
        "passed": train_auc > 0.95,
        "note": "Model should (near-)memorize a tiny slice: train ROC-AUC > 0.95",
    }


def leakage_ceiling(pipeline: Pipeline, X_leaky: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """Demonstrate the dropped leak: with account_status included, AUC ~ 1.0.

    This is why account_status is excluded from the real comparison: it makes
    the task trivially solvable and would prove nothing about churn modeling.
    """
    res = evaluate_model("leaky", pipeline, X_leaky, y, n_splits=n_splits)
    mean_auc, _ = res.mean_sd("roc_auc")
    return {
        "check": "leakage_ceiling",
        "mean_roc_auc": mean_auc,
        "passed": mean_auc > 0.98,
        "note": "Including account_status yields near-perfect AUC -> confirmed leak, correctly dropped",
    }
