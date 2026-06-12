"""Model comparison and sanity checks for the churn experiment.

The single variable across arms is the classifier. Both arms share identical
folds, identical features, identical preprocessing, and fixed seeds, so any
difference in the metric is attributable to the model and not to the harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
N_SPLITS = 5


def make_models(seed: int = SEED) -> dict[str, Pipeline]:
    """The two arms. StandardScaler is fit per-fold on TRAIN only (in-pipeline).

    Scaling is essential for LogisticRegression and harmless for the tree
    ensemble; keeping the pipeline identical holds preprocessing fixed across
    arms. ``class_weight='balanced'`` for LR acknowledges the ~27% positive
    rate; GradientBoosting has no such option, so both are evaluated with
    threshold-free metrics (ROC-AUC, PR-AUC) that do not depend on a cutoff.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=seed),
                ),
            ]
        ),
    }


def _score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


@dataclass
class ArmResult:
    name: str
    per_fold: list[dict[str, float]] = field(default_factory=list)

    def metric_array(self, metric: str) -> np.ndarray:
        return np.array([f[metric] for f in self.per_fold], dtype=float)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for metric in ("roc_auc", "pr_auc"):
            vals = self.metric_array(metric)
            out[metric] = {
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
        return out


def evaluate_arms(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Pipeline],
    n_splits: int = N_SPLITS,
) -> dict[str, ArmResult]:
    """Run every model over the SAME TimeSeriesSplit folds (paired by fold).

    TimeSeriesSplit always trains on past, tests on future relative to the
    time-ordered rows, so it respects the temporal nature of churn while giving
    ``n_splits`` paired estimates per arm for variance.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = {name: ArmResult(name=name) for name in models}
    Xv, yv = X.values, y.values
    for train_idx, test_idx in splitter.split(Xv):
        X_tr, X_te = Xv[train_idx], Xv[test_idx]
        y_tr, y_te = yv[train_idx], yv[test_idx]
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            results[name].per_fold.append(_score(y_te, proba))
    return results


def paired_delta(a: ArmResult, b: ArmResult, metric: str = "roc_auc") -> dict[str, float]:
    """Per-fold paired difference (b - a) with a paired t-test.

    With n_splits folds this is a small sample; we report the mean delta, its sd,
    and the two-sided p-value so the conclusion can be stated honestly rather
    than from a single point estimate.
    """
    from scipy import stats

    da = a.metric_array(metric)
    db = b.metric_array(metric)
    diff = db - da
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t, p = stats.ttest_rel(db, da)
        t, p = float(t), float(p)
    else:
        t, p = 0.0, 1.0
    return {
        "metric": metric,
        "compare": f"{b.name} - {a.name}",
        "mean_delta": mean,
        "sd_delta": sd,
        "n": n,
        "t_stat": t,
        "p_value": p,
    }


# --------------------------------------------------------------------------- #
# Sanity checks: cheap, run before believing any comparison.
# --------------------------------------------------------------------------- #
def sanity_majority_baseline(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """A majority-class predictor must sit at ROC-AUC ~= 0.5. Models must beat it."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    Xv, yv = X.values, y.values
    aucs = []
    for tr, te in splitter.split(Xv):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(Xv[tr], yv[tr])
        proba = dummy.predict_proba(Xv[te])[:, 1]
        # constant prediction -> AUC undefined-ish; roc_auc returns 0.5 for ties
        aucs.append(float(roc_auc_score(yv[te], proba)))
    return {"roc_auc_mean": float(np.mean(aucs))}


def sanity_label_shuffle(
    X: pd.DataFrame, y: pd.Series, seed: int = SEED
) -> dict[str, float]:
    """Shuffle the labels: a leak-free pipeline must collapse to ROC-AUC ~= 0.5."""
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(rng.permutation(y.values), index=y.index)
    model = make_models(seed)["logistic_regression"]
    res = evaluate_arms(X, y_shuf, {"shuffled": model})["shuffled"]
    return {"roc_auc_mean": float(res.metric_array("roc_auc").mean())}


def sanity_overfit_tiny(
    X: pd.DataFrame, y: pd.Series, n: int = 60, seed: int = SEED
) -> dict[str, float]:
    """The model must (over)fit a tiny slice: high TRAIN AUC, else the pipeline is broken."""
    # take a tiny slice that contains both classes
    pos = y[y == 1].index[: n // 2]
    neg = y[y == 0].index[: n // 2]
    idx = list(pos) + list(neg)
    Xs, ys = X.loc[idx], y.loc[idx]
    model = make_models(seed)["gradient_boosting"]
    model.fit(Xs.values, ys.values)
    proba = model.predict_proba(Xs.values)[:, 1]
    return {"train_roc_auc": float(roc_auc_score(ys.values, proba))}


def sanity_leakage_ceiling(
    X_leak: pd.DataFrame, y: pd.Series, seed: int = SEED
) -> dict[str, float]:
    """With the leaked account_status included, AUC should be ~1.0.

    This is the evidence that account_status is a leak and must be dropped --
    not an assertion in our heads.
    """
    model = make_models(seed)["logistic_regression"]
    res = evaluate_arms(X_leak, y, {"with_leak": model})["with_leak"]
    return {"roc_auc_mean": float(res.metric_array("roc_auc").mean())}
