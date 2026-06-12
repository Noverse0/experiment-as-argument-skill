"""The experiment: a time-respecting, leakage-free comparison of LR vs GBM.

Design (one variable: the model family):
  - Variable .......... LogisticRegression  vs  GradientBoostingClassifier.
  - Held fixed ........ identical features, identical CV folds, identical seed,
                        same metric set. Each model is the library default with
                        a fixed random_state (no per-model tuning -> equal
                        tuning budget = zero).
  - Split policy ...... TimeSeriesSplit over date-ordered rows. Every fold trains
                        only on rows that come *before* its test rows, so the
                        evaluation never leaks the future into the past. All
                        preprocessing (scaling) is fit inside a Pipeline on the
                        fold's training portion only -> split-before-transform.
  - Repetition ........ n_splits folds give n paired estimates per model; we
                        report mean +/- sd and the paired per-fold difference.
  - Metric ............ ROC-AUC primary (threshold-free, survives the 27%
                        class imbalance); PR-AUC, Brier, accuracy reported too.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 17
N_SPLITS = 5


def make_models(seed: int = SEED) -> dict:
    """Return the two arms. LR is scaled (it needs it); GBM is scale-invariant.

    Both use library defaults + a fixed seed: the comparison is between model
    *families* at equal (zero) tuning budget, not between hand-tuned variants.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("clf", GradientBoostingClassifier(random_state=seed))]
        ),
    }


def _score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    roc_auc: float
    pr_auc: float
    brier: float
    accuracy: float


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS):
    """Time-ordered CV. Returns per-fold metrics. model is cloned per fold."""
    from sklearn.base import clone

    tscv = TimeSeriesSplit(n_splits=n_splits)
    Xv, yv = X.to_numpy(), y.to_numpy()
    folds = []
    for i, (tr, te) in enumerate(tscv.split(Xv)):
        est = clone(model)
        est.fit(Xv[tr], yv[tr])
        proba = est.predict_proba(Xv[te])[:, 1]
        s = _score(yv[te], proba)
        folds.append(FoldMetrics(fold=i, n_train=len(tr), n_test=len(te), **s))
    return folds


def summarize(folds: list[FoldMetrics]) -> dict:
    """mean / sd / n for each metric across folds."""
    keys = ["roc_auc", "pr_auc", "brier", "accuracy"]
    out = {"n": len(folds)}
    for k in keys:
        vals = np.array([getattr(f, k) for f in folds], dtype=float)
        out[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                  "values": [float(v) for v in vals]}
    return out


def run_comparison(X: pd.DataFrame, y: pd.Series, seed: int = SEED,
                   n_splits: int = N_SPLITS) -> dict:
    """Evaluate both arms on the SAME time folds and compute the paired diff."""
    models = make_models(seed)
    per_model = {}
    raw_folds = {}
    for name, model in models.items():
        folds = evaluate_model(model, X, y, n_splits=n_splits)
        raw_folds[name] = folds
        per_model[name] = {
            "summary": summarize(folds),
            "folds": [asdict(f) for f in folds],
        }

    # Paired per-fold difference on the primary metric (GBM - LR).
    lr = np.array([f.roc_auc for f in raw_folds["logistic_regression"]])
    gb = np.array([f.roc_auc for f in raw_folds["gradient_boosting"]])
    diff = gb - lr
    paired = {
        "metric": "roc_auc",
        "definition": "gradient_boosting - logistic_regression, per fold",
        "per_fold": [float(d) for d in diff],
        "mean": float(diff.mean()),
        "sd": float(diff.std(ddof=1)),
        "n": int(len(diff)),
        "ci95_halfwidth": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
    }
    return {"models": per_model, "paired_diff_roc_auc": paired}


# --------------------------------------------------------------------------- #
# Sanity checks: cheap guards run before trusting the comparison.
# --------------------------------------------------------------------------- #

def sanity_baseline_floor(X, y, seed: int = SEED) -> dict:
    """A no-signal majority/prior classifier. AUC must sit near 0.5."""
    folds = evaluate_model(
        Pipeline([("clf", DummyClassifier(strategy="prior", random_state=seed))]),
        X, y,
    )
    aucs = [f.roc_auc for f in folds]
    return {"mean_roc_auc": float(np.mean(aucs)),
            "passes": bool(abs(np.mean(aucs) - 0.5) < 0.05)}


def sanity_label_shuffle(X, y, seed: int = SEED) -> dict:
    """Shuffle labels: real model's AUC must collapse to ~0.5 (no leakage)."""
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    folds = evaluate_model(make_models(seed)["logistic_regression"], X, y_shuf)
    mean_auc = float(np.mean([f.roc_auc for f in folds]))
    return {"mean_roc_auc": mean_auc, "passes": bool(abs(mean_auc - 0.5) < 0.07)}


def sanity_leakage_ceiling(csv_path: str, seed: int = SEED) -> dict:
    """With the leaky account_status included, AUC should be ~1.0.

    This demonstrates *why* we drop it: it makes the task trivially perfect,
    which on a noisy churn process is the signature of leakage.
    """
    from churn.data import load_with_leak

    X, y = load_with_leak(csv_path)
    folds = evaluate_model(make_models(seed)["logistic_regression"], X, y)
    mean_auc = float(np.mean([f.roc_auc for f in folds]))
    return {"mean_roc_auc_with_leak": mean_auc, "is_near_perfect": bool(mean_auc > 0.99)}


def run_sanity_checks(X, y, csv_path: str, seed: int = SEED) -> dict:
    return {
        "baseline_floor": sanity_baseline_floor(X, y, seed),
        "label_shuffle": sanity_label_shuffle(X, y, seed),
        "leakage_ceiling": sanity_leakage_ceiling(csv_path, seed),
    }
