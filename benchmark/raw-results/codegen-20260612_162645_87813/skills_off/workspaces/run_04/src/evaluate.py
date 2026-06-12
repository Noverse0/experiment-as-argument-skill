"""Evaluation: forward-looking cross-validation, metrics, and sanity checks.

Methodology rationale
---------------------
The task ("predict churn") is forward-looking, and the dataset carries a real
time axis (signup_date, 2023-01 .. 2025-06). A random split would let the model
train on customers who signed up *after* the ones it is scored on -> temporal
leakage. We therefore use sklearn's TimeSeriesSplit on date-sorted rows: every
fold trains on the past and is scored on a strictly later block. The repeated
folds also give us a spread (mean +/- sd, n=folds) so comparative claims carry
variance instead of resting on a single anecdotal split.

Metrics: ROC-AUC and average precision (PR-AUC). The target rate is ~27%, so
accuracy alone is misleading; both reported metrics are threshold-free and
survive class imbalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import features_and_target
from .pipeline import ARMS

N_SPLITS = 5


@dataclass
class ArmResult:
    name: str
    roc_auc: list[float] = field(default_factory=list)
    avg_precision: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        roc = np.array(self.roc_auc)
        ap = np.array(self.avg_precision)
        return {
            "name": self.name,
            "n_folds": len(self.roc_auc),
            "roc_auc_mean": float(roc.mean()),
            "roc_auc_sd": float(roc.std(ddof=1)) if len(roc) > 1 else 0.0,
            "roc_auc_per_fold": [float(x) for x in roc],
            "avg_precision_mean": float(ap.mean()),
            "avg_precision_sd": float(ap.std(ddof=1)) if len(ap) > 1 else 0.0,
            "avg_precision_per_fold": [float(x) for x in ap],
        }


def forward_cv(df, seed: int, n_splits: int = N_SPLITS) -> dict[str, ArmResult]:
    """Run both arms over identical forward-looking folds.

    Both arms see exactly the same train/test indices on every fold (paired
    comparison), so any difference is attributable to the estimator, not to
    luckier splits.
    """
    X, y = features_and_target(df)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = {name: ArmResult(name=name) for name in ARMS}

    for train_idx, test_idx in splitter.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        for name, factory in ARMS.items():
            model = factory(seed)
            model.fit(X_tr, y_tr)  # scaler re-fit on THIS train fold only
            proba = model.predict_proba(X_te)[:, 1]
            results[name].roc_auc.append(roc_auc_score(y_te, proba))
            results[name].avg_precision.append(average_precision_score(y_te, proba))

    return results


def paired_difference(arm_a: ArmResult, arm_b: ArmResult) -> dict:
    """Per-fold paired difference (a - b) on ROC-AUC, with its spread.

    n=N_SPLITS folds is small, so we report mean and sd of the per-fold gap and
    let the reader judge overlap rather than overstating significance.
    """
    a = np.array(arm_a.roc_auc)
    b = np.array(arm_b.roc_auc)
    diff = a - b
    return {
        "metric": "roc_auc",
        "arm_a": arm_a.name,
        "arm_b": arm_b.name,
        "per_fold_diff": [float(x) for x in diff],
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# Sanity checks. These run before we believe any comparative number. Each
# returns a plain dict so run_experiment.py can persist and assert on them.
# --------------------------------------------------------------------------


def baseline_floor(df, seed: int) -> dict:
    """A no-skill classifier must sit at ROC-AUC ~= 0.5. Our models must beat it."""
    X, y = features_and_target(df)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for tr, te in splitter.split(X):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X.iloc[tr], y.iloc[tr])
        proba = dummy.predict_proba(X.iloc[te])[:, 1]
        # prior strategy emits a constant score; AUC is 0.5 by definition.
        aucs.append(roc_auc_score(y.iloc[te], proba))
    return {"check": "baseline_floor", "roc_auc_mean": float(np.mean(aucs))}


def leakage_ceiling(raw_df, seed: int) -> dict:
    """Demonstrate WHY account_status was dropped.

    Fit logreg on the leak column alone. If AUC is near-perfect, the feature
    encodes the target and would have manufactured a fake 'win'. This is the
    evidence behind the drop decision in data.clean_churn.
    """
    from sklearn.linear_model import LogisticRegression

    leak = (raw_df["account_status"] == "closed").astype(int).to_numpy().reshape(-1, 1)
    y = raw_df["churned"].astype(int).to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(leak, y)
    proba = clf.predict_proba(leak)[:, 1]
    return {"check": "leakage_ceiling", "roc_auc": float(roc_auc_score(y, proba))}


def overfit_tiny_subset(df, seed: int) -> dict:
    """A capable model must memorise a tiny slice (train AUC ~= 1.0).

    If it cannot drive train error to ~0 on 60 rows, the pipeline is broken.
    """
    X, y = features_and_target(df)
    tiny = 60
    X_t, y_t = X.iloc[:tiny], y.iloc[:tiny]
    model = ARMS["gboost"](seed)
    model.fit(X_t, y_t)
    proba = model.predict_proba(X_t)[:, 1]
    return {"check": "overfit_tiny_subset", "train_roc_auc": float(roc_auc_score(y_t, proba))}


def label_shuffle(df, seed: int) -> dict:
    """With labels shuffled, signal must vanish (test AUC ~= 0.5).

    If a shuffled-label model still scores well, information is leaking around
    the labels (e.g. through preprocessing or a residual leak feature).
    """
    X, y = features_and_target(df)
    rng = np.random.default_rng(seed)
    y_shuf = y.to_numpy().copy()
    rng.shuffle(y_shuf)

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for tr, te in splitter.split(X):
        model = ARMS["gboost"](seed)
        model.fit(X.iloc[tr], y_shuf[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y_shuf[te], proba))
    return {"check": "label_shuffle", "roc_auc_mean": float(np.mean(aucs))}
