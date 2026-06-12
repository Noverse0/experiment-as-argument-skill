"""The churn model comparison: LogisticRegression vs GradientBoosting.

Methodology (one variable: the classifier):
- Evaluation: ``TimeSeriesSplit`` (forward chaining) on time-ordered rows, so
  every fold trains on earlier signups and tests on later ones. This respects
  the forward-looking nature of churn and gives n folds of variance per arm.
- Preprocessing lives INSIDE a Pipeline, so the scaler is fit on each training
  fold only and applied to the test fold -- never fit on data it will score.
- Metrics: ROC AUC (primary; robust to the ~27% class imbalance) and average
  precision (PR AUC). Accuracy is reported only for context.
- Both arms share the same folds, same features, same seed; only the estimator
  differs. A paired comparison across folds is therefore meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PreparedData

SEED = 7
N_SPLITS = 5


def make_estimators(seed: int = SEED) -> dict[str, Pipeline]:
    """Build the arms. Scaler is harmless for trees and required for LR;
    keeping both in a pipeline guarantees per-fold fitting either way."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


@dataclass
class ArmResult:
    name: str
    roc_auc: list[float] = field(default_factory=list)
    avg_precision: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def ms(xs: list[float]) -> dict:
            arr = np.asarray(xs, dtype=float)
            # ddof=1 sample sd; sd is 0/NaN-safe for n>=2.
            return {
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
                "values": [float(v) for v in arr],
            }

        return {
            "name": self.name,
            "roc_auc": ms(self.roc_auc),
            "avg_precision": ms(self.avg_precision),
            "accuracy": ms(self.accuracy),
        }


def _score_fold(estimator, X_tr, y_tr, X_te, y_te) -> tuple[float, float, float]:
    estimator.fit(X_tr, y_tr)
    proba = estimator.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return (
        roc_auc_score(y_te, proba),
        average_precision_score(y_te, proba),
        float((pred == y_te.to_numpy()).mean()),
    )


def evaluate(
    data: PreparedData,
    seed: int = SEED,
    n_splits: int = N_SPLITS,
    estimators: dict[str, Pipeline] | None = None,
) -> dict[str, ArmResult]:
    """Run every arm + a Dummy baseline through the same time-series folds."""
    estimators = estimators if estimators is not None else make_estimators(seed)
    arms: dict[str, ArmResult] = {
        name: ArmResult(name=name) for name in [*estimators, "baseline_majority"]
    }
    splitter = TimeSeriesSplit(n_splits=n_splits)

    X, y = data.X, data.y
    for tr_idx, te_idx in splitter.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        for name, est in estimators.items():
            roc, ap, acc = _score_fold(est, X_tr, y_tr, X_te, y_te)
            arms[name].roc_auc.append(roc)
            arms[name].avg_precision.append(ap)
            arms[name].accuracy.append(acc)

        # Trivial floor: predicts the training-majority class / prior.
        dummy = DummyClassifier(strategy="prior")
        roc, ap, acc = _score_fold(dummy, X_tr, y_tr, X_te, y_te)
        arms["baseline_majority"].roc_auc.append(roc)
        arms["baseline_majority"].avg_precision.append(ap)
        arms["baseline_majority"].accuracy.append(acc)

    return arms


def label_shuffle_auc(data: PreparedData, seed: int = SEED) -> float:
    """Sanity check: with permuted labels, ROC AUC must collapse to ~0.5.
    If it does not, information is leaking around the labels."""
    rng = np.random.default_rng(seed)
    y_shuf = data.y.to_numpy().copy()
    rng.shuffle(y_shuf)
    est = make_estimators(seed)["logistic_regression"]
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    y_series = data.y.copy()
    y_series[:] = y_shuf
    for tr_idx, te_idx in splitter.split(data.X):
        roc, _, _ = _score_fold(
            est,
            data.X.iloc[tr_idx],
            y_series.iloc[tr_idx],
            data.X.iloc[te_idx],
            y_series.iloc[te_idx],
        )
        aucs.append(roc)
    return float(np.mean(aucs))
