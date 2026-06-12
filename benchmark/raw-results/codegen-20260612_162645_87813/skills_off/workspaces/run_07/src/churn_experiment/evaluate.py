"""Evaluation: forward-looking cross-validation, metrics, and sanity checks.

The comparison is paired across time folds: every fold trains both models on
the same past window and scores them on the same future window, so the
per-fold difference (gb - lr) isolates the model effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

METRICS = ("roc_auc", "pr_auc", "f1")


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def score(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """ROC-AUC and PR-AUC (threshold-free, imbalance-aware) plus F1 at 0.5."""
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


@dataclass
class CVResult:
    model_name: str
    per_fold: list[dict[str, float]] = field(default_factory=list)

    def mean_std(self) -> dict[str, dict[str, float]]:
        out = {}
        for m in METRICS:
            vals = np.array([f[m] for f in self.per_fold], dtype=float)
            out[m] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "values": [float(v) for v in vals],
            }
        return out


def time_series_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    shuffle_labels: bool = False,
    shuffle_seed: int = 0,
) -> CVResult:
    """Run forward-chaining CV. ``model_factory`` returns a fresh estimator.

    Data must already be time-sorted. ``TimeSeriesSplit`` guarantees every test
    fold lies strictly after its training window. With ``shuffle_labels`` the
    training labels are permuted per fold (label-shuffle sanity check); a model
    learning real signal must collapse to the baseline floor here.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    result = CVResult(model_name=getattr(model_factory, "name", "model"))
    rng = np.random.default_rng(shuffle_seed)
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
        if shuffle_labels:
            ytr = rng.permutation(ytr)
        est = clone(model_factory())
        est.fit(Xtr, ytr)
        result.per_fold.append(score(yte, _proba(est, Xte)))
    return result


def baseline_cv(X: pd.DataFrame, y: pd.Series, n_splits: int) -> CVResult:
    """Majority/prior baseline: the floor every model must clear (AUC ~0.5)."""

    def factory():
        return DummyClassifier(strategy="prior")

    factory.name = "baseline_prior"
    return time_series_cv(factory, X, y, n_splits)


def paired_difference(a: CVResult, b: CVResult, metric: str = "roc_auc") -> dict:
    """Paired (b - a) difference across folds with a paired t-test.

    Returns mean diff, sd, n, t-statistic and a two-sided p-value. With small
    n (few folds) this is a weak test by design; the report treats an interval
    that crosses zero as "no detectable difference".
    """
    da = np.array([f[metric] for f in a.per_fold], dtype=float)
    db = np.array([f[metric] for f in b.per_fold], dtype=float)
    diff = db - da
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    t_stat, p_value = _paired_t(diff)
    return {
        "metric": metric,
        "comparison": f"{b.model_name} - {a.model_name}",
        "mean_diff": mean,
        "std_diff": sd,
        "n_folds": n,
        "t_stat": t_stat,
        "p_value": p_value,
        "crosses_zero": bool(abs(mean) <= sd),
    }


def _paired_t(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test without SciPy (Student-t survival via series)."""
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    mean = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0:
        return (float("inf") if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    t = mean / (sd / math.sqrt(n))
    p = _t_sf_two_sided(abs(t), n - 1)
    return float(t), float(p)


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided tail probability of Student-t via the regularized incomplete
    beta function (numerically stable, no SciPy dependency)."""
    x = df / (df + t * t)
    return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's algorithm for the continued fraction.
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        cd = c * d
        f *= cd
        if abs(1.0 - cd) < 1e-10:
            break
    return front * (f - 1.0)
