"""Time-aware evaluation and the paired model comparison.

We use TimeSeriesSplit on the signup-ordered data: every fold trains on earlier
customers and tests on later ones, which matches a forward-looking churn task and
avoids the leakage of a random split. The folds also give us repetition — n
measurements per arm — so a comparative claim can carry a mean ± sd, not a single
anecdote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import Dataset, assert_no_leak_columns

# Metrics chosen to survive class imbalance: ROC-AUC and PR-AUC are threshold-free
# and do not collapse to "predict the majority". Accuracy is reported only
# alongside the majority-class baseline so it cannot masquerade as skill.
PRIMARY_METRIC = "roc_auc"


@dataclass
class ArmResult:
    name: str
    per_fold: dict[str, list[float]] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        return float(np.mean(self.per_fold[metric]))

    def sd(self, metric: str) -> float:
        # Sample sd (ddof=1); reported as the spread across folds.
        vals = self.per_fold[metric]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def n(self) -> int:
        return len(next(iter(self.per_fold.values())))


def _fold_metrics(y_true, y_prob, y_pred) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(np.mean(y_pred == y_true)),
        # Majority-class baseline accuracy on THIS fold's test set: the floor any
        # real model must clear.
        "baseline_accuracy": float(max(np.mean(y_true), 1 - np.mean(y_true))),
    }


def evaluate_model(
    model, dataset: Dataset, n_splits: int = 5
) -> ArmResult:
    """Run time-series CV for one model and collect per-fold metrics."""
    X = dataset.X
    y = dataset.y.to_numpy()
    assert_no_leak_columns(X)  # guard: leak columns must never reach the model

    splitter = TimeSeriesSplit(n_splits=n_splits)
    result = ArmResult(name=getattr(model, "name", "model"))
    metric_keys = ["roc_auc", "pr_auc", "accuracy", "baseline_accuracy"]
    result.per_fold = {k: [] for k in metric_keys}

    X_arr = X.to_numpy()
    for train_idx, test_idx in splitter.split(X_arr):
        est = clone(model)
        est.fit(X_arr[train_idx], y[train_idx])
        prob = est.predict_proba(X_arr[test_idx])[:, 1]
        pred = est.predict(X_arr[test_idx])
        m = _fold_metrics(y[test_idx], prob, pred)
        for k in metric_keys:
            result.per_fold[k].append(m[k])

    return result


@dataclass
class Comparison:
    arm_a: ArmResult  # baseline arm (logistic regression)
    arm_b: ArmResult  # challenger arm (gradient boosting)
    metric: str
    mean_diff: float  # arm_b - arm_a
    sd_diff: float
    t_stat: float
    p_value: float
    conclusion: str


def compare_models(
    baseline: ArmResult, challenger: ArmResult, metric: str = PRIMARY_METRIC
) -> Comparison:
    """Paired comparison of challenger vs baseline across folds.

    The folds are paired (same train/test indices for both arms), so a paired
    t-test on the per-fold differences is the right test. With a small number of
    folds this is underpowered; we therefore phrase the conclusion conservatively
    and only claim a winner when the difference clears noise.
    """
    a = np.array(baseline.per_fold[metric])
    b = np.array(challenger.per_fold[metric])
    diffs = b - a
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # Paired t-test. If every difference is identical (sd 0) scipy returns nan;
    # treat that as "no measurable difference".
    if sd_diff == 0.0:
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_rel(b, a)
        t_stat, p_value = float(t_stat), float(p_value)

    alpha = 0.05
    if p_value < alpha and mean_diff > 0:
        conclusion = (
            f"gradient_boosting outperforms logistic_regression on {metric} "
            f"(mean diff {mean_diff:+.4f}, p={p_value:.3f}, n={len(diffs)})"
        )
    elif p_value < alpha and mean_diff < 0:
        conclusion = (
            f"logistic_regression outperforms gradient_boosting on {metric} "
            f"(mean diff {mean_diff:+.4f}, p={p_value:.3f}, n={len(diffs)})"
        )
    else:
        conclusion = (
            f"no detectable difference on {metric}: mean diff {mean_diff:+.4f} "
            f"(sd {sd_diff:.4f}, p={p_value:.3f}, n={len(diffs)}); the gap is "
            f"within fold-to-fold noise"
        )

    return Comparison(
        arm_a=baseline,
        arm_b=challenger,
        metric=metric,
        mean_diff=mean_diff,
        sd_diff=sd_diff,
        t_stat=t_stat,
        p_value=p_value,
        conclusion=conclusion,
    )
