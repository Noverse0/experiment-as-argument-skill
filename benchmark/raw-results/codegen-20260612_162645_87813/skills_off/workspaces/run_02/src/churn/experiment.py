"""Evaluation methodology and sanity checks.

Methodology (see REPORT.md for the rationale):

  * Time-based evaluation. Rows are ordered by signup_date and fed to
    ``TimeSeriesSplit`` (expanding window). Every test fold is strictly later
    than the data the model trained on, matching the forward-looking nature of
    churn prediction. This yields ``n_splits`` measurements per model, so the
    comparison reports mean +/- sd over folds rather than a single anecdote.
  * Preprocessing is fitted per fold on the training portion only (the scaler
    lives inside the Pipeline), so no test statistics leak into fitting.
  * Metrics: ROC AUC (primary; robust to the ~27% class imbalance), average
    precision (PR AUC), plus accuracy and F1 for context.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .data import LoadedData
from .models import RANDOM_STATE, model_factories

N_SPLITS = 5
PRIMARY_METRIC = "roc_auc"


def _score(y_true, y_prob, y_pred) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(estimator: Pipeline, X: pd.DataFrame, y: pd.Series,
                   n_splits: int = N_SPLITS) -> list[dict]:
    """Return one metric dict per forward-looking fold."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    for train_idx, test_idx in splitter.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = model.predict(X.iloc[test_idx])
        fold_scores.append(_score(y.iloc[test_idx], y_prob, y_pred))
    return fold_scores


def _aggregate(fold_scores: list[dict]) -> dict:
    metrics = fold_scores[0].keys()
    out = {}
    for m in metrics:
        vals = [fs[m] for fs in fold_scores]
        out[m] = {
            "mean": float(statistics.mean(vals)),
            "sd": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "per_fold": [float(v) for v in vals],
        }
    return out


@dataclass
class ArmResult:
    name: str
    folds: list[dict]
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.summary:
            self.summary = _aggregate(self.folds)


def run_comparison(data: LoadedData, n_splits: int = N_SPLITS) -> dict:
    """Run every model arm and compute the paired comparison on the primary
    metric. Returns a JSON-serializable result dict."""
    arms = {}
    for name, factory in model_factories().items():
        folds = evaluate_model(factory(), data.X, data.y, n_splits=n_splits)
        arms[name] = ArmResult(name=name, folds=folds)

    a, b = "gradient_boosting", "logistic_regression"
    diffs = [
        arms[a].folds[i][PRIMARY_METRIC] - arms[b].folds[i][PRIMARY_METRIC]
        for i in range(n_splits)
    ]
    mean_diff = float(statistics.mean(diffs))
    sd_diff = float(statistics.stdev(diffs)) if len(diffs) > 1 else 0.0

    # Paired t-test as a heuristic only. Expanding-window folds share training
    # data, so they are not independent; we lean on the spread, not the p-value.
    from scipy import stats

    if sd_diff > 0:
        t_stat, p_value = stats.ttest_rel(
            [arms[a].folds[i][PRIMARY_METRIC] for i in range(n_splits)],
            [arms[b].folds[i][PRIMARY_METRIC] for i in range(n_splits)],
        )
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = 0.0, 1.0

    # Honest verdict: only claim a winner if the per-fold paired difference is
    # consistent and clears its own spread (|mean| > sd) AND the heuristic test
    # agrees. Otherwise: no detectable difference.
    decisive = abs(mean_diff) > sd_diff and p_value < 0.05
    if not decisive:
        verdict = "no_detectable_difference"
    elif mean_diff > 0:
        verdict = "gradient_boosting_better"
    else:
        verdict = "logistic_regression_better"

    return {
        "primary_metric": PRIMARY_METRIC,
        "n_splits": n_splits,
        "arms": {name: arm.summary for name, arm in arms.items()},
        "comparison": {
            "definition": f"{a} minus {b} on {PRIMARY_METRIC}, paired per fold",
            "per_fold_diff": [float(d) for d in diffs],
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "paired_t_stat": t_stat,
            "paired_p_value": p_value,
            "verdict": verdict,
        },
    }


# --------------------------------------------------------------------------- #
# Sanity checks: cheap guards that catch silent pipeline bugs and prove the
# leak defenses actually matter. Each returns a JSON-serializable dict.
# --------------------------------------------------------------------------- #

def sanity_baseline_floor(data: LoadedData) -> dict:
    """A no-information baseline must sit at chance (ROC AUC ~ 0.5)."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for tr, te in splitter.split(data.X):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(data.X.iloc[tr], data.y.iloc[tr])
        prob = dummy.predict_proba(data.X.iloc[te])[:, 1]
        # A constant-probability classifier has AUC exactly 0.5 by convention.
        aucs.append(roc_auc_score(data.y.iloc[te], prob))
    mean_auc = float(statistics.mean(aucs))
    return {"mean_roc_auc": mean_auc, "passed": abs(mean_auc - 0.5) < 0.05}


def sanity_label_shuffle(data: LoadedData) -> dict:
    """With shuffled labels, real models must collapse to chance. If they do
    not, information is leaking around the labels."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_shuffled = pd.Series(
        rng.permutation(data.y.to_numpy()), index=data.y.index
    )
    shuffled = LoadedData(
        X=data.X, y=y_shuffled, dates=data.dates,
        n_raw=data.n_raw, n_duplicates_removed=data.n_duplicates_removed,
        churn_rate=float(y_shuffled.mean()),
    )
    from .models import make_gradient_boosting

    folds = evaluate_model(make_gradient_boosting(), shuffled.X, shuffled.y)
    mean_auc = float(statistics.mean(f["roc_auc"] for f in folds))
    return {"mean_roc_auc": mean_auc, "passed": abs(mean_auc - 0.5) < 0.08}


def sanity_leakage_ceiling(csv_path: str) -> dict:
    """Re-introduce account_status as a feature. It should make the task
    near-trivial (AUC ~ 1.0), which is exactly why the real pipeline drops it.
    This guards against silently re-admitting the leak."""
    raw = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    raw["signup_date"] = pd.to_datetime(raw["signup_date"])
    raw = raw.sort_values("signup_date", kind="stable").reset_index(drop=True)
    leak = (raw["account_status"] == "closed").astype(int).to_frame("is_closed")
    y = raw["churned"].astype(int)

    from sklearn.linear_model import LogisticRegression

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for tr, te in splitter.split(leak):
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(leak.iloc[tr], y.iloc[tr])
        prob = clf.predict_proba(leak.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
    mean_auc = float(statistics.mean(aucs))
    return {"mean_roc_auc": mean_auc, "passed": mean_auc > 0.99}


def run_sanity_checks(data: LoadedData, csv_path: str) -> dict:
    return {
        "baseline_floor": sanity_baseline_floor(data),
        "label_shuffle": sanity_label_shuffle(data),
        "leakage_ceiling": sanity_leakage_ceiling(csv_path),
    }
