"""The comparison itself.

Evaluation methodology
----------------------
The task is forward-looking churn prediction and the data carries a temporal
``signup_date``, so a random split would leak the future into the past. We:

1. Order rows by ``signup_date`` and hold out the LAST 20% as a one-time test
   set (touched once, at the very end, for both models).
2. On the first 80% (development set) we compare the two classifiers with
   ``TimeSeriesSplit`` forward-chaining cross-validation: each fold trains on an
   earlier window and validates on the next, never on the past. Folds are
   identical for both arms, so AUC differences are paired.
3. We repeat the whole CV over several seeds. LogisticRegression is
   deterministic; GradientBoosting is not -- repeating exposes its variance.
   Each (seed, fold) pair is one paired measurement.

Metrics are ROC-AUC and average-precision (PR-AUC), both threshold-free and
robust to the 27% class imbalance. We report mean +/- sd with n, and a paired
t-test on the per-fold AUC differences. No winner is declared if the difference
is within noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import PreparedData
from .models import MODEL_FACTORIES

SEEDS = [0, 1, 2]
N_SPLITS = 5
TEST_FRACTION = 0.20


@dataclass
class ArmScores:
    name: str
    cv_roc_auc: list[float]
    cv_pr_auc: list[float]

    def summary(self) -> dict:
        roc = np.array(self.cv_roc_auc)
        pr = np.array(self.cv_pr_auc)
        return {
            "name": self.name,
            "n_cv_measurements": len(roc),
            "roc_auc_mean": float(roc.mean()),
            "roc_auc_sd": float(roc.std(ddof=1)),
            "pr_auc_mean": float(pr.mean()),
            "pr_auc_sd": float(pr.std(ddof=1)),
            "cv_roc_auc": self.cv_roc_auc,
            "cv_pr_auc": self.cv_pr_auc,
        }


def _split_dev_test(data: PreparedData):
    n = len(data.y)
    cut = int(n * (1 - TEST_FRACTION))
    X_dev, X_test = data.X.iloc[:cut], data.X.iloc[cut:]
    y_dev, y_test = data.y.iloc[:cut], data.y.iloc[cut:]
    return X_dev, X_test, y_dev, y_test, cut


def cross_validate_arms(data: PreparedData) -> dict[str, ArmScores]:
    """Forward-chaining CV on the development split, repeated over seeds."""
    X_dev, _, y_dev, _, _ = _split_dev_test(data)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    # Precompute folds once so both arms see identical (paired) folds.
    folds = list(tscv.split(X_dev))

    arms: dict[str, ArmScores] = {
        name: ArmScores(name=name, cv_roc_auc=[], cv_pr_auc=[])
        for name in MODEL_FACTORIES
    }

    for seed in SEEDS:
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx]
            X_va, y_va = X_dev.iloc[va_idx], y_dev.iloc[va_idx]
            for name, factory in MODEL_FACTORIES.items():
                model = factory(seed)
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_va)[:, 1]
                arms[name].cv_roc_auc.append(float(roc_auc_score(y_va, proba)))
                arms[name].cv_pr_auc.append(
                    float(average_precision_score(y_va, proba))
                )
    return arms


def paired_comparison(arms: dict[str, ArmScores]) -> dict:
    """Paired t-test on per-(seed,fold) ROC-AUC differences (GBM - LogReg)."""
    gbm = np.array(arms["gradient_boosting"].cv_roc_auc)
    lr = np.array(arms["logistic_regression"].cv_roc_auc)
    diff = gbm - lr
    # Two-sided paired t-test; folds are paired by construction.
    t_stat, p_value = stats.ttest_rel(gbm, lr)
    return {
        "metric": "roc_auc",
        "delta_mean_gbm_minus_lr": float(diff.mean()),
        "delta_sd": float(diff.std(ddof=1)),
        "n_pairs": int(len(diff)),
        "paired_t_stat": float(t_stat),
        "paired_p_value": float(p_value),
    }


def final_holdout(data: PreparedData) -> dict:
    """Touch the held-out test set ONCE. Train each arm on the full dev split
    (seed 0) and report test ROC-AUC / PR-AUC for both."""
    X_dev, X_test, y_dev, y_test, cut = _split_dev_test(data)
    out = {"test_size": int(len(y_test)), "test_churn_rate": float(y_test.mean())}
    per_model = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory(SEEDS[0])
        model.fit(X_dev, y_dev)
        proba = model.predict_proba(X_test)[:, 1]
        per_model[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        }
    out["models"] = per_model
    return out


def conclude(comparison: dict, alpha: float = 0.05) -> str:
    """Honest conclusion: only claim a winner if the paired test clears alpha."""
    p = comparison["paired_p_value"]
    delta = comparison["delta_mean_gbm_minus_lr"]
    if p >= alpha:
        return (
            "No detectable difference: the ROC-AUC gap between gradient boosting "
            f"and logistic regression (delta={delta:+.4f}) is within noise "
            f"(paired t-test p={p:.3f} >= {alpha})."
        )
    winner = "gradient boosting" if delta > 0 else "logistic regression"
    return (
        f"{winner.capitalize()} outperforms the other arm on ROC-AUC "
        f"(delta={delta:+.4f}, paired t-test p={p:.3f} < {alpha})."
    )


def run_full_experiment(data: PreparedData, path: str) -> dict:
    from .sanity import run_all

    sanity = run_all(data, path, SEEDS[0])
    arms = cross_validate_arms(data)
    comparison = paired_comparison(arms)
    holdout = final_holdout(data)
    return {
        "config": {
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "test_fraction": TEST_FRACTION,
            "cv": "TimeSeriesSplit (forward-chaining)",
            "features": list(data.X.columns),
        },
        "data": {
            "n_raw_rows": data.n_raw,
            "n_duplicates_dropped": data.n_duplicates_dropped,
            "n_rows_used": int(len(data.y)),
            "churn_rate": data.churn_rate,
        },
        "sanity": sanity,
        "arms": {name: arm.summary() for name, arm in arms.items()},
        "comparison": comparison,
        "holdout": holdout,
        "conclusion": conclude(comparison),
    }
