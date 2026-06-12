"""Leakage-aware comparison of LogisticRegression vs GradientBoostingClassifier.

Methodology (see REPORT.md for the rationale):
- Single variable: the classifier. Preprocessing/features/splits/seeds are fixed.
- TimeSeriesSplit (forward-chaining): every test fold is later in signup time than
  its training data -- a forward-looking evaluation, not a random split.
- StandardScaler fit on the training fold only, inside a Pipeline (no leakage).
- Primary metric ROC AUC; also average precision (imbalance-aware), accuracy, F1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PreparedData

SEED = 7
N_SPLITS = 5
PRIMARY_METRIC = "roc_auc"
MODELS = ("logistic_regression", "gradient_boosting")


def make_pipeline(model_name: str, seed: int = SEED) -> Pipeline:
    """Identical preprocessing for both arms; only the final estimator differs."""
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown model: {model_name}")
    # StandardScaler is harmless for trees and necessary for LR convergence;
    # using it for both keeps preprocessing identical across arms.
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _score(y_true, y_pred, y_proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(data: PreparedData, model_name: str, seed: int = SEED,
                   n_splits: int = N_SPLITS) -> list[dict]:
    """Per-fold metrics for one model under forward-chaining time-based CV."""
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    for fold_idx, (tr, te) in enumerate(splitter.split(X)):
        pipe = make_pipeline(model_name, seed=seed)
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = _score(y[te], pred, proba)
        m.update({"fold": fold_idx, "n_train": int(len(tr)), "n_test": int(len(te)),
                  "test_churn_rate": float(y[te].mean())})
        folds.append(m)
    return folds


def _summary(folds: list[dict], metric: str) -> dict:
    vals = np.array([f[metric] for f in folds], dtype=float)
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "n": int(len(vals)), "values": vals.tolist()}


@dataclass
class SanityResults:
    baseline_auc_mean: float  # DummyClassifier; expect ~0.5
    label_shuffle_auc_mean: float  # shuffled labels; expect ~0.5
    leakage_ceiling_auc_mean: float  # with account_status; expect ~1.0
    determinism_ok: bool  # same seed -> identical metrics


def run_sanity_checks(data: PreparedData, leaky_X: np.ndarray,
                      seed: int = SEED, n_splits: int = N_SPLITS) -> SanityResults:
    """Cheap checks that catch silent pipeline bugs before we trust any result."""
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits)

    # Baseline floor: a stratified dummy should sit at AUC ~ 0.5.
    base_aucs = []
    for tr, te in splitter.split(X):
        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        dummy.fit(X[tr], y[tr])
        base_aucs.append(roc_auc_score(y[te], dummy.predict_proba(X[te])[:, 1]))

    # Label-shuffle: destroy the X->y relationship; real model must collapse to ~0.5.
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    shuf_aucs = []
    for tr, te in splitter.split(X):
        pipe = make_pipeline("logistic_regression", seed=seed)
        pipe.fit(X[tr], y_shuf[tr])
        shuf_aucs.append(roc_auc_score(y_shuf[te], pipe.predict_proba(X[te])[:, 1]))

    # Leakage ceiling: account_status alone makes the task ~perfect. This is WHY
    # it is excluded from the real feature set.
    leak_aucs = []
    for tr, te in splitter.split(leaky_X):
        pipe = make_pipeline("logistic_regression", seed=seed)
        pipe.fit(leaky_X[tr], y[tr])
        leak_aucs.append(roc_auc_score(y[te], pipe.predict_proba(leaky_X[te])[:, 1]))

    # Determinism: re-running an arm with the same seed must reproduce metrics exactly.
    a = evaluate_model(data, "gradient_boosting", seed=seed, n_splits=n_splits)
    b = evaluate_model(data, "gradient_boosting", seed=seed, n_splits=n_splits)
    determinism_ok = all(
        abs(x["roc_auc"] - y_["roc_auc"]) < 1e-12 for x, y_ in zip(a, b)
    )

    return SanityResults(
        baseline_auc_mean=float(np.mean(base_aucs)),
        label_shuffle_auc_mean=float(np.mean(shuf_aucs)),
        leakage_ceiling_auc_mean=float(np.mean(leak_aucs)),
        determinism_ok=bool(determinism_ok),
    )


def compare(data: PreparedData, leaky_X: np.ndarray, seed: int = SEED,
            n_splits: int = N_SPLITS) -> dict:
    """Run both arms, summarize per metric, and run a paired comparison on AUC."""
    per_model = {name: evaluate_model(data, name, seed=seed, n_splits=n_splits)
                 for name in MODELS}

    metrics = ["roc_auc", "average_precision", "accuracy", "f1"]
    summaries = {
        name: {m: _summary(folds, m) for m in metrics}
        for name, folds in per_model.items()
    }

    # Paired (per-fold) comparison on the primary metric: GB - LR.
    lr = np.array([f[PRIMARY_METRIC] for f in per_model["logistic_regression"]])
    gb = np.array([f[PRIMARY_METRIC] for f in per_model["gradient_boosting"]])
    diff = gb - lr
    # Paired t-test; n=5 folds, so treat the p-value as a weak signal, not proof.
    if np.allclose(diff, diff[0]):
        t_stat, p_value = float("nan"), float("nan")
    else:
        t_stat, p_value = stats.ttest_rel(gb, lr)

    paired = {
        "metric": PRIMARY_METRIC,
        "gb_minus_lr_mean": float(diff.mean()),
        "gb_minus_lr_sd": float(diff.std(ddof=1)),
        "per_fold_diff": diff.tolist(),
        "paired_t_stat": float(t_stat),
        "paired_p_value": float(p_value),
        "n_folds": int(len(diff)),
    }

    sanity = run_sanity_checks(data, leaky_X, seed=seed, n_splits=n_splits)

    return {
        "config": {
            "seed": seed,
            "n_splits": n_splits,
            "split": "TimeSeriesSplit (forward-chaining, ordered by signup_date)",
            "models": list(MODELS),
            "features": list(data.X.columns),
            "primary_metric": PRIMARY_METRIC,
            "preprocessing": "StandardScaler fit on train fold only",
        },
        "data": {
            "n_raw": data.n_raw,
            "n_after_dedup": data.n_after_dedup,
            "n_duplicates_removed": data.n_duplicates_removed,
            "churn_rate": data.churn_rate,
        },
        "per_fold": per_model,
        "summary": summaries,
        "paired_primary": paired,
        "sanity": asdict(sanity),
    }


def conclusion_text(result: dict) -> str:
    """Honest one-paragraph verdict driven strictly by the measured numbers."""
    paired = result["paired_primary"]
    lr = result["summary"]["logistic_regression"][PRIMARY_METRIC]
    gb = result["summary"]["gradient_boosting"][PRIMARY_METRIC]
    diff_mean, diff_sd = paired["gb_minus_lr_mean"], paired["gb_minus_lr_sd"]
    p = paired["paired_p_value"]

    # "Detectable" = the per-fold diff is more than ~1 sd from zero AND p < 0.05.
    detectable = (
        not np.isnan(p) and p < 0.05 and abs(diff_mean) > diff_sd
    )
    if not detectable:
        verdict = (
            "No detectable difference: gradient boosting does not outperform "
            "logistic regression on this dataset. The per-fold AUC gap "
            f"({diff_mean:+.4f} +/- {diff_sd:.4f}, n={paired['n_folds']}) is within "
            "noise."
        )
    elif diff_mean > 0:
        verdict = (
            f"Gradient boosting outperforms logistic regression on AUC by "
            f"{diff_mean:+.4f} +/- {diff_sd:.4f} (paired p={p:.3f}, n={paired['n_folds']})."
        )
    else:
        verdict = (
            f"Logistic regression outperforms gradient boosting on AUC by "
            f"{-diff_mean:+.4f} +/- {diff_sd:.4f} (paired p={p:.3f}, n={paired['n_folds']})."
        )
    return (
        f"LR AUC = {lr['mean']:.4f} +/- {lr['sd']:.4f}; "
        f"GB AUC = {gb['mean']:.4f} +/- {gb['sd']:.4f} (n={lr['n']} folds). {verdict}"
    )
