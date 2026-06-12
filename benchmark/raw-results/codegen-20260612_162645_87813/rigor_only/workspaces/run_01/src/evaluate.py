"""Time-based evaluation and the paired comparison between arms.

Variance source: TimeSeriesSplit folds. signup_date is temporal and churn is forward-looking,
so a random split would leak the future into the past. We instead use an expanding-window
TimeSeriesSplit: every test fold lies strictly after its training rows in time. The n folds
give us n paired measurements per arm -> mean +/- sd, which is what turns a single number
into evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 5


@dataclass
class ArmResult:
    name: str
    roc_auc: list[float] = field(default_factory=list)
    pr_auc: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def ms(xs):
            a = np.asarray(xs, dtype=float)
            # sample sd (ddof=1); meaningful only with >=2 folds.
            sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            return {"mean": float(a.mean()), "sd": sd, "n": int(len(a)), "values": [float(x) for x in a]}

        return {"roc_auc": ms(self.roc_auc), "pr_auc": ms(self.pr_auc), "brier": ms(self.brier)}


def evaluate_arms(arms: dict, X, y, n_splits: int = N_SPLITS) -> dict[str, ArmResult]:
    """Run every arm through the SAME time-ordered folds. X/y must already be time-sorted."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = {name: ArmResult(name=name) for name in arms}

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for train_idx, test_idx in splitter.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        for name, pipe in arms.items():
            model = clone(pipe)  # fresh fit per fold; scaler learns train-fold stats only
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            results[name].roc_auc.append(roc_auc_score(y_te, proba))
            results[name].pr_auc.append(average_precision_score(y_te, proba))
            results[name].brier.append(brier_score_loss(y_te, proba))
    return results


def paired_comparison(a: ArmResult, b: ArmResult, metric: str = "roc_auc") -> dict:
    """Paired per-fold difference (a - b) on `metric`, with a cautious paired t-test.

    With n=5 folds this is a low-power test; we report it but lean on the spread for the
    honest conclusion rather than chasing a p-value.
    """
    av = np.asarray(getattr(a, metric), dtype=float)
    bv = np.asarray(getattr(b, metric), dtype=float)
    diff = av - bv
    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0

    t_stat = p_value = None
    if n > 1 and sd_diff > 0:
        t_stat = mean_diff / (sd_diff / math.sqrt(n))
        try:
            from scipy import stats  # optional; absent -> p_value stays None

            p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
            t_stat = float(t_stat)
        except Exception:
            t_stat = float(t_stat)

    # "Detectable" only if the mean gap exceeds one combined sd (a deliberately modest bar).
    detectable = abs(mean_diff) > sd_diff and sd_diff > 0

    return {
        "metric": metric,
        "arm_a": a.name,
        "arm_b": b.name,
        "per_fold_diff": [float(d) for d in diff],
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "n": int(n),
        "t_stat": t_stat,
        "p_value": p_value,
        "detectable_difference": bool(detectable),
    }
