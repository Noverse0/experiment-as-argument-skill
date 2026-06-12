"""Model definitions and the comparative evaluation.

The single variable under study is the MODEL (LogisticRegression vs
GradientBoostingClassifier). Everything else is held fixed:
- same features, same cleaning, same folds (paired comparison),
- same preprocessing policy (StandardScaler fit on the train fold only, inside a
  Pipeline; harmless for the tree model, required for LR),
- same seeds.

Primary metric: ROC-AUC (threshold-free, robust to the ~0.27 class imbalance).
Secondary: average precision (PR-AUC) and accuracy. A majority-class baseline is
reported so "beats trivial" is checkable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TIME_COLUMN, clean, features_and_target

N_SPLITS = 5
N_REPEATS = 3
DEFAULT_SEED = 20260612


def make_models(seed: int) -> dict[str, Pipeline]:
    """Return the two competing pipelines. Tuning budget = defaults for both
    (held fixed: neither model gets a hyperparameter search the other lacks)."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),  # harmless for trees; keeps pipelines symmetric
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


@dataclass
class ArmResult:
    name: str
    roc_auc_mean: float
    roc_auc_sd: float
    avg_precision_mean: float
    avg_precision_sd: float
    accuracy_mean: float
    accuracy_sd: float
    n_estimates: int
    roc_auc_per_fold: list[float]


def _evaluate_arm(name: str, pipe: Pipeline, X, y, cv) -> ArmResult:
    aucs, aps, accs = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xtr, ytr)  # preprocessing fit on train fold only (inside pipeline)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = pipe.predict(Xte)
        aucs.append(roc_auc_score(yte, proba))
        aps.append(average_precision_score(yte, proba))
        accs.append((pred == yte.values).mean())
    return ArmResult(
        name=name,
        roc_auc_mean=float(np.mean(aucs)),
        roc_auc_sd=float(np.std(aucs, ddof=1)),
        avg_precision_mean=float(np.mean(aps)),
        avg_precision_sd=float(np.std(aps, ddof=1)),
        accuracy_mean=float(np.mean(accs)),
        accuracy_sd=float(np.std(accs, ddof=1)),
        n_estimates=len(aucs),
        roc_auc_per_fold=[float(a) for a in aucs],
    )


def majority_baseline_auc(X, y, cv) -> dict:
    """Trivial baseline. AUC of a constant predictor is ~0.5 by construction."""
    aucs, accs = [], []
    for train_idx, test_idx in cv.split(X, y):
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X.iloc[train_idx], ytr)
        proba = dummy.predict_proba(X.iloc[test_idx])[:, 1]
        # constant proba -> roc_auc is undefined-ish; sklearn returns 0.5
        aucs.append(roc_auc_score(yte, proba))
        accs.append((dummy.predict(X.iloc[test_idx]) == yte.values).mean())
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "n_estimates": len(aucs),
    }


def paired_comparison(arm_a: ArmResult, arm_b: ArmResult) -> dict:
    """Paired comparison of per-fold ROC-AUC (same folds for both arms).

    Returns mean difference, its 95% CI, and a paired t-test. The honest claim
    depends on whether the CI excludes 0, not on the point estimate alone.
    """
    a = np.array(arm_a.roc_auc_per_fold)
    b = np.array(arm_b.roc_auc_per_fold)
    diff = b - a  # positive => arm_b (gradient boosting) higher
    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se = sd_diff / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
    tstat, pval = stats.ttest_rel(b, a)
    return {
        "arm_a": arm_a.name,
        "arm_b": arm_b.name,
        "metric": "roc_auc",
        "mean_diff_b_minus_a": mean_diff,
        "sd_diff": sd_diff,
        "ci95_diff": [float(ci[0]), float(ci[1])],
        "t_statistic": float(tstat),
        "p_value": float(pval),
        "n_pairs": int(n),
        "significant_at_0.05": bool(pval < 0.05),
    }


def run_full_experiment(df_raw: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Clean -> CV evaluate both arms on identical folds -> paired comparison."""
    df = clean(df_raw)
    X, y = features_and_target(df, include_leak=False)

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed
    )
    arms = {
        name: _evaluate_arm(name, pipe, X, y, cv)
        for name, pipe in make_models(seed).items()
    }
    baseline = majority_baseline_auc(X, y, cv)
    comparison = paired_comparison(
        arms["logistic_regression"], arms["gradient_boosting"]
    )
    return {
        "config": {
            "seed": seed,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "n_estimates": N_SPLITS * N_REPEATS,
            "features": list(X.columns),
            "metric_primary": "roc_auc",
        },
        "majority_baseline": baseline,
        "arms": {name: asdict(res) for name, res in arms.items()},
        "comparison": comparison,
    }


def time_based_split_eval(df_raw: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Robustness check: respect time. Train on the earliest 80% of signups,
    test on the latest 20%. One split -> one number per arm, no variance, so this
    is a directional cross-check on the CV conclusion, not the primary evidence."""
    df = clean(df_raw)  # already sorted by signup_date
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    Xtr, ytr = features_and_target(train_df, include_leak=False)
    Xte, yte = features_and_target(test_df, include_leak=False)

    out = {
        "split_time": str(pd.to_datetime(test_df[TIME_COLUMN]).min().date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_base_rate": float(ytr.mean()),
        "test_base_rate": float(yte.mean()),
        "arms": {},
    }
    for name, pipe in make_models(seed).items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        out["arms"][name] = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "avg_precision": float(average_precision_score(yte, proba)),
        }
    return out
