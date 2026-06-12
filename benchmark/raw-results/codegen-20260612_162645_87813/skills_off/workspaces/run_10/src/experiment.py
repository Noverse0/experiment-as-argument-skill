"""The churn comparison experiment: GradientBoosting vs LogisticRegression.

Evaluation methodology (justified in REPORT.md):
- Time-based cross-validation (TimeSeriesSplit) on signup_date order, because
  the task is forward-looking and the data carries a temporal column.
- No per-model hyperparameter tuning, so every fold's validation split is
  legitimately out-of-sample; the CV mean +/- sd over folds is the comparison
  statistic. Models are paired by fold for an honest per-fold gap.
- Preprocessing (StandardScaler) is fit on the training fold only, inside a
  Pipeline, so no test-fold statistics leak into fitting.
- Primary metric is ROC AUC (threshold-free, survives the ~27% imbalance);
  average precision and accuracy are reported for context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CleanData

SEED = 7
N_SPLITS = 5


def make_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Build the two arms. Scaler is included for both for a fair pipeline.

    LogisticRegression is deterministic; GradientBoosting's randomness is
    pinned via random_state so re-runs are identical.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


@dataclass
class ArmResult:
    name: str
    roc_auc_mean: float
    roc_auc_sd: float
    roc_auc_per_fold: list[float]
    average_precision_mean: float
    average_precision_sd: float
    accuracy_mean: float
    n_folds: int


def _fit_eval(model: Pipeline, X, y, train_idx, test_idx) -> dict[str, float]:
    """Fit on train fold, score on test fold. Returns metrics dict."""
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = y.iloc[test_idx]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float((pred == y_test.values).mean()),
    }


def evaluate_arm(
    name: str, model: Pipeline, X, y, splits
) -> tuple[ArmResult, list[float]]:
    """Run one model across all CV folds. Returns the summary and per-fold AUC."""
    per_fold = [_fit_eval(model, X, y, tr, te) for tr, te in splits]
    auc = [m["roc_auc"] for m in per_fold]
    ap = [m["average_precision"] for m in per_fold]
    acc = [m["accuracy"] for m in per_fold]
    result = ArmResult(
        name=name,
        roc_auc_mean=float(np.mean(auc)),
        roc_auc_sd=float(np.std(auc, ddof=1)),
        roc_auc_per_fold=[float(a) for a in auc],
        average_precision_mean=float(np.mean(ap)),
        average_precision_sd=float(np.std(ap, ddof=1)),
        accuracy_mean=float(np.mean(acc)),
        n_folds=len(per_fold),
    )
    return result, auc


def baseline_floor_auc(data: CleanData, splits) -> float:
    """DummyClassifier(prior) AUC across folds. Should be ~0.5; models beat it."""
    dummy = DummyClassifier(strategy="prior")
    aucs = []
    for tr, te in splits:
        dummy.fit(data.X.iloc[tr], data.y.iloc[tr])
        # Prior dummy gives a constant score; AUC is 0.5 by definition but we
        # compute it for an honest, non-hardcoded floor.
        proba = dummy.predict_proba(data.X.iloc[te])[:, 1]
        y_test = data.y.iloc[te]
        try:
            aucs.append(float(roc_auc_score(y_test, proba)))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))


def label_shuffle_auc(data: CleanData, splits, seed: int = SEED) -> float:
    """AUC of the GBM arm trained on SHUFFLED labels. Should collapse to ~0.5.

    If it does not, information is leaking around the labels.
    """
    rng = np.random.default_rng(seed)
    y_shuf = data.y.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    model = make_models(seed)["gradient_boosting"]
    aucs = []
    for tr, te in splits:
        model.fit(data.X.iloc[tr], y_shuf.iloc[tr])
        proba = model.predict_proba(data.X.iloc[te])[:, 1]
        aucs.append(float(roc_auc_score(y_shuf.iloc[te], proba)))
    return float(np.mean(aucs))


def leakage_ceiling_auc(X_leaky, y, splits, seed: int = SEED) -> float:
    """AUC when the leak column is INCLUDED. Should be ~1.0, justifying the drop."""
    model = make_models(seed)["gradient_boosting"]
    aucs = []
    for tr, te in splits:
        model.fit(X_leaky.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X_leaky.iloc[te])[:, 1]
        aucs.append(float(roc_auc_score(y.iloc[te], proba)))
    return float(np.mean(aucs))


@dataclass
class Comparison:
    """Paired per-fold comparison of GBM minus LogReg AUC."""

    gbm_minus_logreg_mean: float
    gbm_minus_logreg_sd: float
    per_fold_diff: list[float]
    detectable_difference: bool  # does the +/-1sd band exclude 0?
    conclusion: str


def compare(gbm_auc: list[float], logreg_auc: list[float]) -> Comparison:
    """Honest paired comparison. A winner is claimed only if the per-fold
    difference's mean +/- 1 sd band excludes zero."""
    diff = np.array(gbm_auc) - np.array(logreg_auc)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    lo, hi = mean - sd, mean + sd
    detectable = (lo > 0) or (hi < 0)
    if not detectable:
        conclusion = (
            "No detectable difference: the per-fold AUC gap "
            f"({mean:+.4f} +/- {sd:.4f}) overlaps zero across {len(diff)} folds."
        )
    elif mean > 0:
        conclusion = (
            f"Gradient boosting outperforms logistic regression by "
            f"{mean:+.4f} AUC (+/-{sd:.4f}) across {len(diff)} folds."
        )
    else:
        conclusion = (
            f"Logistic regression outperforms gradient boosting by "
            f"{-mean:.4f} AUC (+/-{sd:.4f}) across {len(diff)} folds."
        )
    return Comparison(
        gbm_minus_logreg_mean=mean,
        gbm_minus_logreg_sd=sd,
        per_fold_diff=[float(d) for d in diff],
        detectable_difference=detectable,
        conclusion=conclusion,
    )


def run(data: CleanData, leaky_X, seed: int = SEED, n_splits: int = N_SPLITS) -> dict:
    """Run the full experiment and return a JSON-serializable result dict."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(data.X))

    models = make_models(seed)
    arms: dict[str, ArmResult] = {}
    auc_by_arm: dict[str, list[float]] = {}
    for name, model in models.items():
        arms[name], auc_by_arm[name] = evaluate_arm(
            name, model, data.X, data.y, splits
        )

    sanity = {
        "baseline_floor_auc": baseline_floor_auc(data, splits),
        "label_shuffle_auc": label_shuffle_auc(data, splits, seed),
        "leakage_ceiling_auc": leakage_ceiling_auc(leaky_X, data.y, splits, seed),
    }

    comparison = compare(
        auc_by_arm["gradient_boosting"], auc_by_arm["logistic_regression"]
    )

    return {
        "config": {
            "seed": seed,
            "n_splits": n_splits,
            "cv": "TimeSeriesSplit (forward-chaining on signup_date order)",
            "features": list(data.X.columns),
            "primary_metric": "roc_auc",
            "sklearn_models": {
                "logistic_regression": "LogisticRegression(max_iter=1000)",
                "gradient_boosting": "GradientBoostingClassifier(defaults)",
            },
        },
        "data": {
            "n_raw": data.n_raw,
            "n_after_dedup": int(len(data.X)),
            "n_duplicates_removed": data.n_duplicates_removed,
            "churn_rate": data.churn_rate,
        },
        "arms": {name: asdict(res) for name, res in arms.items()},
        "sanity_checks": sanity,
        "comparison": asdict(comparison),
    }
