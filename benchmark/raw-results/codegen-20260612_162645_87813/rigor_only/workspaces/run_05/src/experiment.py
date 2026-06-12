"""The churn model comparison and its sanity checks.

Claim under test: for predicting ``churned`` on this dataset, does
``GradientBoostingClassifier`` outperform ``LogisticRegression``?

Design:
- Single variable: the classifier. Everything else (features, split, folds,
  preprocessing policy, seed) is held fixed across the two arms.
- Split policy: time-ordered ``TimeSeriesSplit`` (train on past, evaluate on
  future), respecting the temporal ``signup_date`` and giving N folds of
  variance per arm. The test portion of each fold is scored once.
- Preprocessing: fit per-fold on training rows only, inside a Pipeline, so no
  statistic from a fold's evaluation rows reaches the fit (split-before-transform).
- Metrics: ROC-AUC (primary; survives the 27% class imbalance), average
  precision (PR-AUC), and accuracy (reported alongside the majority-class
  baseline so accuracy is interpretable, not celebrated on its own).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PreparedData

SEED = 7
N_SPLITS = 5


def make_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Build the two arms. Same seed; LogReg is scaled, GBM is not (trees are
    scale-invariant). Both are deterministic given the seed."""
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    gbm = Pipeline(
        steps=[
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]
    )
    return {"logistic_regression": logreg, "gradient_boosting": gbm}


def _score(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def _agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    # population-free sample sd (ddof=1); folds are the repeats.
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "sd": sd, "n": int(arr.size)}


@dataclass
class ArmResult:
    name: str
    per_fold: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        metrics = self.per_fold[0].keys()
        return {m: _agg([f[m] for f in self.per_fold]) for m in metrics}


def evaluate_arms(
    data: PreparedData, seed: int = SEED, n_splits: int = N_SPLITS
) -> dict:
    """Run both arms and the majority baseline through the same time folds.

    Returns a JSON-serializable dict with per-fold scores, per-arm summaries,
    the paired ROC-AUC difference (GBM - LogReg) across folds, and the baseline.
    """
    X = data.X.to_numpy()
    y = data.y.to_numpy()

    splitter = TimeSeriesSplit(n_splits=n_splits)
    models = make_models(seed)
    arms = {name: ArmResult(name) for name in models}
    baseline = ArmResult("majority_baseline")

    fold_indices = list(splitter.split(X))
    for train_idx, test_idx in fold_indices:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Majority-class baseline floor (predicts the train-majority label).
        dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        baseline.per_fold.append(
            _score(y_te, dummy.predict_proba(X_te)[:, 1], dummy.predict(X_te))
        )

        for name, model in models.items():
            # Clone-free: refit a fresh pipeline each fold to avoid leakage of
            # state across folds. make_models() returns fresh estimators, so we
            # rebuild per fold.
            fresh = make_models(seed)[name]
            fresh.fit(X_tr, y_tr)
            proba = fresh.predict_proba(X_te)[:, 1]
            pred = fresh.predict(X_te)
            arms[name].per_fold.append(_score(y_te, proba, pred))

    # Paired difference on the primary metric, per fold (GBM - LogReg).
    # Folds are paired (same train/test rows feed both arms), so a paired test
    # on the per-fold differences is the right comparison.
    gbm_auc = [f["roc_auc"] for f in arms["gradient_boosting"].per_fold]
    lr_auc = [f["roc_auc"] for f in arms["logistic_regression"].per_fold]
    diffs = [g - l for g, l in zip(gbm_auc, lr_auc)]
    paired = _paired_test(diffs)

    return {
        "seed": seed,
        "n_splits": n_splits,
        "fold_sizes": [
            {"train": int(len(tr)), "test": int(len(te))} for tr, te in fold_indices
        ],
        "arms": {
            name: {"per_fold": arm.per_fold, "summary": arm.summary()}
            for name, arm in arms.items()
        },
        "majority_baseline": {
            "per_fold": baseline.per_fold,
            "summary": baseline.summary(),
        },
        "paired_roc_auc_diff_gbm_minus_lr": {
            "per_fold": diffs,
            **_agg(diffs),
            **paired,
        },
    }


def _paired_test(diffs: list[float], alpha: float = 0.05) -> dict:
    """Two-sided paired t-test on the per-fold differences.

    Caveat recorded for the report: the folds of an expanding-window
    TimeSeriesSplit are not fully independent (training windows overlap), so
    this p-value is approximate and slightly anti-conservative. It is used as a
    guardrail against over-claiming, not as a precise inference.
    """
    from scipy import stats

    arr = np.asarray(diffs, dtype=float)
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return {"t_statistic": 0.0, "p_value": 1.0, "alpha": alpha, "significant": False}
    t, p = stats.ttest_1samp(arr, 0.0)
    return {
        "t_statistic": float(t),
        "p_value": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
    }


# --------------------------------------------------------------------------- #
# Sanity checks (run before believing the comparison).
# --------------------------------------------------------------------------- #


def sanity_leakage_ceiling(csv_path: str, seed: int = SEED) -> dict:
    """If we KEEP the leaky account_status, AUC should jump to ~1.0.

    Confirms the dropped column was genuinely a leak and worth dropping.
    """
    from .data import with_leak_feature

    X, y = with_leak_feature(csv_path)
    X, y = X.to_numpy(), y.to_numpy()
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for train_idx, test_idx in splitter.split(X):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
    return {"mean_roc_auc_with_leak": float(np.mean(aucs))}


def sanity_label_shuffle(data: PreparedData, seed: int = SEED) -> dict:
    """With labels shuffled, AUC must collapse to ~0.5. If it does not, the
    features are leaking the target around the labels."""
    rng = np.random.default_rng(seed)
    X = data.X.to_numpy()
    y = data.y.to_numpy().copy()
    rng.shuffle(y)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for train_idx, test_idx in splitter.split(X):
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
    return {"mean_roc_auc_shuffled_labels": float(np.mean(aucs))}


def sanity_overfit_tiny(data: PreparedData, seed: int = SEED) -> dict:
    """A flexible model must (near-)perfectly fit a tiny slice it trained on.
    If it cannot memorize 50 rows, the pipeline is broken."""
    X = data.X.to_numpy()[:50]
    y = data.y.to_numpy()[:50]
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X, y)
    return {"train_accuracy_tiny_subset": float(accuracy_score(y, clf.predict(X)))}
