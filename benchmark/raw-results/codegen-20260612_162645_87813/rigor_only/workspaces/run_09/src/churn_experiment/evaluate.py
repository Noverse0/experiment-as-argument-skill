"""Time-ordered cross-validated comparison of the two arms.

Evaluation methodology and its justification:

- **Split:** ``TimeSeriesSplit`` (expanding window). Churn is forward-looking and
  the data carries a real ``signup_date``, so every test fold lies strictly
  *after* its training rows in time. A random split would leak the future.
- **Repeats:** the 5 time folds give 5 paired measurements per arm (CV folds
  count as repeats). We never claim a winner from a single split.
- **Seeds:** GradientBoosting is stochastic; we re-run the whole fold sweep over
  3 seeds to confirm the conclusion is not a seed artefact. LogisticRegression
  (liblinear) is deterministic.
- **Metrics:** ROC-AUC (primary; threshold-free, robust to the 27% imbalance)
  and Average Precision / PR-AUC (positive-class focus under imbalance). Both
  are compared against trivial baselines (AUC 0.5; AP = prevalence).
- **Test contact:** each fold's test rows are scored exactly once per
  (arm, seed, fold); no tuning decision is taken after seeing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import ChurnData
from .pipeline import ARMS, make_pipeline

N_SPLITS = 5
SEEDS = (0, 1, 2)


@dataclass
class FoldResult:
    arm: str
    seed: int
    fold: int
    roc_auc: float
    avg_precision: float


@dataclass
class ArmSummary:
    arm: str
    roc_auc_mean: float
    roc_auc_sd: float
    avg_precision_mean: float
    avg_precision_sd: float
    n: int


@dataclass
class Comparison:
    """Paired GradientBoosting - LogisticRegression difference per (seed, fold)."""

    metric: str
    mean_diff: float
    sd_diff: float
    n_pairs: int
    diffs: list = field(default_factory=list)


def run_folds(data: ChurnData, seeds=SEEDS, n_splits=N_SPLITS) -> list[FoldResult]:
    """Fit every arm on every (seed, time-fold) and record held-out metrics."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    X, y = data.X, data.y
    results: list[FoldResult] = []
    for seed in seeds:
        for fold, (tr, te) in enumerate(splitter.split(X)):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            for arm in ARMS:
                pipe = make_pipeline(arm, seed)
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_te)[:, 1]
                results.append(
                    FoldResult(
                        arm=arm,
                        seed=seed,
                        fold=fold,
                        roc_auc=float(roc_auc_score(y_te, proba)),
                        avg_precision=float(average_precision_score(y_te, proba)),
                    )
                )
    return results


def summarise(results: list[FoldResult]) -> dict[str, ArmSummary]:
    out: dict[str, ArmSummary] = {}
    for arm in ARMS:
        roc = np.array([r.roc_auc for r in results if r.arm == arm])
        ap = np.array([r.avg_precision for r in results if r.arm == arm])
        out[arm] = ArmSummary(
            arm=arm,
            roc_auc_mean=float(roc.mean()),
            roc_auc_sd=float(roc.std(ddof=1)),
            avg_precision_mean=float(ap.mean()),
            avg_precision_sd=float(ap.std(ddof=1)),
            n=int(roc.size),
        )
    return out


def compare(results: list[FoldResult], metric: str = "roc_auc") -> Comparison:
    """Paired difference (gboost - logreg) on matched (seed, fold) cells."""
    by_key = {}
    for r in results:
        by_key.setdefault((r.seed, r.fold), {})[r.arm] = getattr(r, metric)
    diffs = [
        cell["gboost"] - cell["logreg"]
        for cell in by_key.values()
        if "gboost" in cell and "logreg" in cell
    ]
    arr = np.array(diffs)
    return Comparison(
        metric=metric,
        mean_diff=float(arr.mean()),
        sd_diff=float(arr.std(ddof=1)),
        n_pairs=int(arr.size),
        diffs=[float(d) for d in diffs],
    )


def verdict(comp: Comparison) -> str:
    """Honest conclusion: a winner only if the paired spread clears zero.

    With n paired folds we use the mean +/- 1 sd band as a coarse uncertainty
    interval. If that band straddles zero the honest claim is "no detectable
    difference" rather than naming a winner.
    """
    lo = comp.mean_diff - comp.sd_diff
    hi = comp.mean_diff + comp.sd_diff
    if lo > 0:
        return "gboost"
    if hi < 0:
        return "logreg"
    return "no detectable difference"
