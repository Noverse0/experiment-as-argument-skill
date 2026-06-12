"""Entrypoint: run the full churn GBM-vs-LogReg experiment.

Usage:
    python3 make_dataset.py --out churn.csv   # once, to create the data
    python3 run_experiment.py                 # runs everything

Writes:
    results/metrics.json   machine-readable: config, seeds, sanity checks, per-fold + aggregate
    REPORT.md              human-readable conclusion, methodology, limitations
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.data import FEATURES, ID_COLS, LEAK_COLS, load_clean, split_xy
from src.experiment import (
    N_SPLITS,
    SEED,
    aggregate,
    baseline_auc,
    evaluate_model,
    label_shuffle_auc,
    overfit_tiny_auc,
)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
CSV = ROOT / "churn.csv"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run() -> dict:
    if not CSV.exists():
        raise SystemExit("churn.csv not found. Run: python3 make_dataset.py --out churn.csv")

    df, stats = load_clean(str(CSV))
    X, y = split_xy(df, include_leak=False)

    # --- Sanity checks (must pass before the comparison means anything) ----- #
    sanity = {
        "majority_baseline_roc_auc": baseline_auc(X, y),
        "label_shuffle_roc_auc": {
            "logreg": label_shuffle_auc("logreg", X, y),
            "gbm": label_shuffle_auc("gbm", X, y),
        },
        "overfit_tiny_train_roc_auc": {
            "logreg": overfit_tiny_auc("logreg", X, y),
            "gbm": overfit_tiny_auc("gbm", X, y),
        },
        # Leak probe: if we (wrongly) include account_status, AUC should be ~1.0,
        # which is exactly why the real comparison drops it.
        "leak_probe_with_account_status_roc_auc": float(
            np.mean(
                [
                    f.roc_auc
                    for f in evaluate_model(
                        "logreg", *split_xy(df, include_leak=True)
                    )
                ]
            )
        ),
    }

    # --- The comparison ----------------------------------------------------- #
    logreg_folds = evaluate_model("logreg", X, y)
    gbm_folds = evaluate_model("gbm", X, y)
    logreg_agg = aggregate(logreg_folds)
    gbm_agg = aggregate(gbm_folds)

    # Paired comparison on ROC-AUC across the same folds.
    lr_auc = np.array([f.roc_auc for f in logreg_folds])
    gbm_auc = np.array([f.roc_auc for f in gbm_folds])
    diff = gbm_auc - lr_auc  # positive => GBM better
    try:
        wstat, pval = wilcoxon(gbm_auc, lr_auc)
        wstat, pval = float(wstat), float(pval)
    except ValueError:
        wstat, pval = float("nan"), float("nan")  # all-equal differences

    comparison = {
        "metric": "roc_auc",
        "mean_diff_gbm_minus_logreg": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "per_fold_diff": diff.tolist(),
        "wilcoxon_stat": wstat,
        "wilcoxon_pvalue": pval,
    }

    metrics = {
        "config": {
            "models": ["logreg", "gbm"],
            "seed": SEED,
            "n_splits": N_SPLITS,
            "cv": "TimeSeriesSplit (forward-chaining, sorted by signup_date)",
            "features": FEATURES,
            "dropped_leak_cols": list(LEAK_COLS),
            "dropped_id_cols": list(ID_COLS),
            "primary_metric": "roc_auc",
            "secondary_metrics": ["pr_auc", "brier"],
        },
        "environment": {
            "python": platform.python_version(),
            "git_rev": _git_rev(),
            "dataset_command": "python3 make_dataset.py --out churn.csv",
        },
        "data": {
            "n_raw": stats.n_raw,
            "n_exact_duplicates_removed": stats.n_exact_duplicates,
            "n_after_dedup": stats.n_after_dedup,
            "churn_rate": stats.churn_rate,
        },
        "sanity": sanity,
        "results": {"logreg": logreg_agg, "gbm": gbm_agg},
        "comparison": comparison,
    }
    return metrics


def conclusion(m: dict) -> str:
    lr = m["results"]["logreg"]
    gb = m["results"]["gbm"]
    d = m["comparison"]
    lr_m, lr_s = lr["roc_auc_mean"], lr["roc_auc_sd"]
    gb_m, gb_s = gb["roc_auc_mean"], gb["roc_auc_sd"]
    # "Detectable" = means differ by more than the combined fold-to-fold spread.
    overlap = abs(gb_m - lr_m) <= (lr_s + gb_s)
    if overlap:
        return (
            f"**No detectable difference.** GBM ROC-AUC {gb_m:.3f} +/- {gb_s:.3f} vs "
            f"LogReg {lr_m:.3f} +/- {lr_s:.3f} (n={lr['n_folds']} time folds). The "
            f"mean gap ({d['mean_diff_gbm_minus_logreg']:+.3f}) is within the "
            f"fold-to-fold spread, so this experiment does not support a winner."
        )
    better = "GBM" if gb_m > lr_m else "LogReg"
    return (
        f"**{better} scores higher.** GBM {gb_m:.3f} +/- {gb_s:.3f} vs LogReg "
        f"{lr_m:.3f} +/- {lr_s:.3f} (n={lr['n_folds']}), mean gap "
        f"{d['mean_diff_gbm_minus_logreg']:+.3f} (GBM-LogReg), Wilcoxon "
        f"p={d['wilcoxon_pvalue']:.3f}. Treat as suggestive given n={lr['n_folds']}."
    )


def render_report(m: dict) -> str:
    data, cfg, san = m["data"], m["config"], m["sanity"]
    lr, gb = m["results"]["logreg"], m["results"]["gbm"]

    def row(agg):
        return (
            f"{agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_sd']:.3f} | "
            f"{agg['pr_auc_mean']:.3f} +/- {agg['pr_auc_sd']:.3f} | "
            f"{agg['brier_mean']:.3f} +/- {agg['brier_sd']:.3f}"
        )

    return f"""# Churn: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset?

## Conclusion
{conclusion(m)}

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| LogReg | {row(lr)} |
| GBM    | {row(gb)} |

Reported as mean +/- sd across n={lr['n_folds']} forward-chaining time folds.
Paired difference (GBM-LogReg) on ROC-AUC: {m['comparison']['mean_diff_gbm_minus_logreg']:+.3f} +/- {m['comparison']['sd_diff']:.3f}, Wilcoxon p={m['comparison']['wilcoxon_pvalue']:.3f}.

## Methodology
- **Single variable:** the classifier. Both arms share identical features,
  preprocessing, splits, and seed ({cfg['seed']}). The only difference is the estimator.
- **Features used:** {cfg['features']}.
- **Dropped — target leakage:** {cfg['dropped_leak_cols']}. `account_status` is
  `"closed"` iff `churned==1` (confirmed: 0 mislabeled rows), so it encodes the
  label and was removed. Leak probe below quantifies the effect.
- **Dropped — identifier:** {cfg['dropped_id_cols']}.
- **Deduplication:** removed {data['n_exact_duplicates_removed']} exact duplicate
  rows ({data['n_raw']} -> {data['n_after_dedup']}) **before** splitting, so no
  customer straddles the train/test boundary.
- **Split:** forward-chaining `TimeSeriesSplit` ({cfg['n_splits']} folds) over rows
  sorted by `signup_date`. `signup_date` is temporal and the task is
  forward-looking, so a random split would leak the future; it is used only to
  order rows, never as a feature.
- **Preprocessing:** `StandardScaler` fit on the training fold only
  (split-before-transform), applied to the test fold.
- **Class balance:** churn rate = {data['churn_rate']:.3f} (imbalanced), so the
  primary metric is ROC-AUC with PR-AUC and Brier alongside, not accuracy.

## Sanity checks (run before trusting the comparison)
- **Majority baseline ROC-AUC:** {san['majority_baseline_roc_auc']:.3f} (≈0.5 expected — models must beat this).
- **Label-shuffle ROC-AUC:** logreg {san['label_shuffle_roc_auc']['logreg']:.3f}, gbm {san['label_shuffle_roc_auc']['gbm']:.3f} (≈0.5 expected — confirms no information leaks around the labels).
- **Overfit tiny slice (train AUC):** logreg {san['overfit_tiny_train_roc_auc']['logreg']:.3f}, gbm {san['overfit_tiny_train_roc_auc']['gbm']:.3f} (near 1.0 — pipeline can learn).
- **Leak probe (account_status included):** ROC-AUC {san['leak_probe_with_account_status_roc_auc']:.3f} (≈1.0 — demonstrates why the column is dropped).

## Limitations
- n={lr['n_folds']} time folds is a small sample; folds have different train sizes
  and test windows, so they are not i.i.d. repeats. Variance estimates are rough.
- The dataset is synthetic with a near-linear log-odds structure, which plays to a
  linear model's strengths and gives GBM little non-linearity to exploit; this likely
  explains the dead heat. Conclusions may not transfer to real churn data.
- The test windows are scored once per fold; no hyperparameter tuning was done on
  them. Models use scikit-learn defaults, so this compares default-configured
  estimators, not tuned ones.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```
Seed={cfg['seed']}, Python {m['environment']['python']}, git {m['environment']['git_rev']}.
"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    m = run()
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=2))
    (ROOT / "REPORT.md").write_text(render_report(m))
    print(conclusion(m).replace("**", ""))
    print(f"\nWrote {RESULTS / 'metrics.json'} and {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
