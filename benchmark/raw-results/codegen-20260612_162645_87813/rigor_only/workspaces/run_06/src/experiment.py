"""The churn comparison experiment: models, time-based CV, and sanity checks.

Design (one variable: the classifier):
* Preprocessing is identical for both arms -- a StandardScaler fit on the
  training fold only (split-before-transform). GBM does not need scaling but it
  is harmless, and keeping the pipeline identical means the *only* thing that
  differs between arms is the estimator.
* Evaluation is forward-chaining ``TimeSeriesSplit`` over rows sorted by
  signup date: every fold trains on the past and tests on the future. This both
  respects the temporal nature of the data and gives n paired measurements per
  arm so we can report mean +/- sd rather than a single anecdote.
* Metrics: ROC-AUC and PR-AUC (average precision) because the target is
  imbalanced (~27% positive); Brier score for calibration. Accuracy is reported
  only alongside a majority-class baseline so it is never read in isolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
N_SPLITS = 5


def make_model(name: str, seed: int = SEED) -> Pipeline:
    """Build a pipeline: train-fold-fitted scaler + the chosen classifier.

    A plain StandardScaler scales every column of the X passed to ``fit`` (all
    features here are numeric), so the leak-probe arm that adds an extra column
    is actually scaled and seen by the model.
    """
    pre = StandardScaler()
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "gbm":
        clf = GradientBoostingClassifier(random_state=seed)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("pre", pre), ("clf", clf)])


def _score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    roc_auc: float
    pr_auc: float
    brier: float


def evaluate_model(name: str, X, y, n_splits: int = N_SPLITS, seed: int = SEED) -> list[FoldResult]:
    """Forward-chaining time-CV. X/y must already be time-ordered and deduped."""
    X = X.reset_index(drop=True)
    y = np.asarray(y).ravel()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out: list[FoldResult] = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        model = make_model(name, seed=seed)
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        s = _score(y[te], proba)
        out.append(
            FoldResult(
                fold=i, n_train=len(tr), n_test=len(te),
                roc_auc=s["roc_auc"], pr_auc=s["pr_auc"], brier=s["brier"],
            )
        )
    return out


def aggregate(folds: list[FoldResult]) -> dict:
    """Mean +/- sd (population-free sample sd, ddof=1) across folds, with n."""
    arr = {k: np.array([getattr(f, k) for f in folds]) for k in ("roc_auc", "pr_auc", "brier")}
    agg = {"n_folds": len(folds), "folds": [asdict(f) for f in folds]}
    for k, v in arr.items():
        agg[f"{k}_mean"] = float(v.mean())
        agg[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return agg


# --------------------------------------------------------------------------- #
# Sanity checks -- run before believing the comparison.
# --------------------------------------------------------------------------- #
def baseline_auc(X, y, n_splits: int = N_SPLITS, seed: int = SEED) -> float:
    """Majority/stratified DummyClassifier floor. Must sit near 0.5 ROC-AUC."""
    X = X.reset_index(drop=True)
    y = np.asarray(y).ravel()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr, te in tscv.split(X):
        d = DummyClassifier(strategy="stratified", random_state=seed)
        d.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], d.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def label_shuffle_auc(name: str, X, y, seed: int = SEED) -> float:
    """Train on shuffled labels. A clean pipeline must collapse to ~0.5 AUC."""
    rng = np.random.default_rng(seed)
    y_shuf = np.asarray(y).ravel().copy()
    rng.shuffle(y_shuf)
    folds = evaluate_model(name, X, y_shuf, seed=seed)
    return float(np.mean([f.roc_auc for f in folds]))


def overfit_tiny_auc(name: str, X, y, n: int = 60, seed: int = SEED) -> float:
    """Model must (over)fit a tiny slice: train-set AUC should be near 1.0."""
    Xs = X.reset_index(drop=True).iloc[:n]
    ys = np.asarray(y).ravel()[:n]
    model = make_model(name, seed=seed)
    model.fit(Xs, ys)
    return float(roc_auc_score(ys, model.predict_proba(Xs)[:, 1]))
