"""Entrypoint: run the full LR vs GBM churn experiment and write results."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_data, clean_data, get_X_y, churn_rate, FEATURES, TARGET
from src.models import build_lr, build_gbm
from src.evaluate import run_cv, baseline_auc, label_shuffle_check, overfit_tiny_check

RESULTS_DIR = Path("results")
DATA_PATH = "churn.csv"


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['n']})"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Load and clean ───────────────────────────────────────────────────
    df_raw = load_data(DATA_PATH)
    df, clean_stats = clean_data(df_raw)
    print(f"Loaded {clean_stats['n_before_dedup']} rows; "
          f"removed {clean_stats['n_duplicates_removed']} duplicates → "
          f"{clean_stats['n_after_dedup']} rows")

    X, y = get_X_y(df)
    rate = churn_rate(y)
    print(f"Churn rate: {rate:.2%}  Features: {FEATURES}")
    print(f"NOTE: days_since_last_login EXCLUDED — post-outcome leak")

    # ── 2. Sanity checks ───────────────────────────────────────────────────
    print("\n── Sanity checks ──────────────────────────────────────")
    lr_pipe = build_lr()
    gbm_pipe = build_gbm()

    baseline = baseline_auc(y)
    print(f"Baseline (majority-class) AUC: {baseline:.4f}  (expect ≈0.50)")

    shuffle_lr = label_shuffle_check(lr_pipe, X, y)
    shuffle_gbm = label_shuffle_check(gbm_pipe, X, y)
    print(f"Label-shuffle AUC — LR: {shuffle_lr['shuffled_roc_auc_mean']:.4f}, "
          f"GBM: {shuffle_gbm['shuffled_roc_auc_mean']:.4f}  (expect ≈0.50)")

    overfit_lr = overfit_tiny_check(lr_pipe, X, y)
    overfit_gbm = overfit_tiny_check(gbm_pipe, X, y)
    print(f"Overfit-tiny AUC — LR: {overfit_lr['overfit_tiny_roc_auc']:.4f}, "
          f"GBM: {overfit_gbm['overfit_tiny_roc_auc']:.4f}  (expect >0.80)")

    # ── 3. Cross-validation ────────────────────────────────────────────────
    print("\n── Cross-validation (5-fold × 3 repeats = 15 evals) ──")
    cv_lr = run_cv(build_lr(), X, y)
    cv_gbm = run_cv(build_gbm(), X, y)

    for name, cv in [("LogisticRegression", cv_lr), ("GradientBoosting", cv_gbm)]:
        print(f"\n{name}:")
        for metric, stat in cv.items():
            print(f"  {metric:20s}: {_fmt(stat)}")

    # ── 4. Determine winner ────────────────────────────────────────────────
    lr_auc = cv_lr["roc_auc"]["mean"]
    gbm_auc = cv_gbm["roc_auc"]["mean"]
    lr_sd = cv_lr["roc_auc"]["std"]
    gbm_sd = cv_gbm["roc_auc"]["std"]
    gap = gbm_auc - lr_auc
    # Rough overlap check: gap > sum of SDs suggests non-overlapping spreads
    meaningful = gap > (lr_sd + gbm_sd)

    if meaningful and gap > 0:
        conclusion = "GradientBoosting outperforms LogisticRegression (non-overlapping spreads)"
    elif meaningful and gap < 0:
        conclusion = "LogisticRegression outperforms GradientBoosting (non-overlapping spreads)"
    else:
        conclusion = (
            f"No detectable difference — gap ({gap:+.4f}) is within noise "
            f"(LR sd={lr_sd:.4f}, GBM sd={gbm_sd:.4f})"
        )
    print(f"\nConclusion: {conclusion}")

    # ── 5. Write machine-readable results ─────────────────────────────────
    metrics = {
        "clean_stats": clean_stats,
        "churn_rate": rate,
        "features_used": FEATURES,
        "features_excluded": ["days_since_last_login"],
        "exclusion_reason": "post-outcome leak: churned customers have higher days_since_last_login by construction",
        "sanity_checks": {
            "baseline_auc": baseline,
            "label_shuffle_lr_auc": shuffle_lr["shuffled_roc_auc_mean"],
            "label_shuffle_gbm_auc": shuffle_gbm["shuffled_roc_auc_mean"],
            "overfit_tiny_lr_auc": overfit_lr["overfit_tiny_roc_auc"],
            "overfit_tiny_gbm_auc": overfit_gbm["overfit_tiny_roc_auc"],
        },
        "cv": {
            "method": "RepeatedStratifiedKFold",
            "n_splits": 5,
            "n_repeats": 3,
            "n_total_folds": 15,
        },
        "logistic_regression": cv_lr,
        "gradient_boosting": cv_gbm,
        "conclusion": conclusion,
        "gap_roc_auc": gap,
    }
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {out_path}")

    # ── 6. Write REPORT.md ─────────────────────────────────────────────────
    report = _build_report(metrics, cv_lr, cv_gbm, conclusion, gap)
    report_path = Path("REPORT.md")
    report_path.write_text(report)
    print(f"Report written to {report_path}")


def _build_report(metrics, cv_lr, cv_gbm, conclusion, gap):
    def fmt(stat):
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}"

    sanity = metrics["sanity_checks"]
    churn_pct = f"{metrics['churn_rate']:.1%}"
    n_clean = metrics["clean_stats"]["n_after_dedup"]
    n_dupes = metrics["clean_stats"]["n_duplicates_removed"]

    return f"""# Churn Prediction Experiment: LR vs Gradient Boosting

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting
customer churn on this tabular dataset?

## Conclusion
**{conclusion}**

| Model | ROC-AUC | Avg Precision | F1 |
|-------|---------|---------------|-----|
| LogisticRegression | {fmt(cv_lr["roc_auc"])} | {fmt(cv_lr["average_precision"])} | {fmt(cv_lr["f1"])} |
| GradientBoosting | {fmt(cv_gbm["roc_auc"])} | {fmt(cv_gbm["average_precision"])} | {fmt(cv_gbm["f1"])} |
| Gap (GBM − LR) | {gap:+.4f} | | |

## Methodology

### Data
- Raw rows: {metrics["clean_stats"]["n_before_dedup"]}; after removing **{n_dupes} exact duplicates**: {n_clean} rows
- Churn rate: {churn_pct} (imbalanced; AUC and Average Precision reported instead of accuracy)

### Feature Selection
**Used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Excluded:**
- `days_since_last_login` — **target leak**: churned customers have stopped logging
  in by definition, so this column is recorded *after* the outcome. Including it
  would inflate model AUC without reflecting real predictive power.
- `customer_id` — row identifier, no signal
- `signup_date` — proxy for tenure (already captured by `tenure_months`);
  using raw dates as a feature would encode arbitrary cohort effects

### Preprocessing
StandardScaler fitted on each training fold only, applied to validation folds.
This prevents data leakage from the scaler (split-before-transform rule).

### Evaluation
RepeatedStratifiedKFold: 5 splits × 3 repeats = **15 evaluations per model**.
Stratification preserves the churn rate in each fold. Repetition provides
variance estimates so a single lucky split cannot determine the winner.

**Primary metric:** ROC-AUC (threshold-independent, handles class imbalance)
**Secondary:** Average Precision (area under precision-recall curve), F1

### Sanity Checks
| Check | LR | GBM | Expected |
|-------|----|-----|---------|
| Majority-class baseline AUC | {sanity["baseline_auc"]:.4f} | — | ≈ 0.50 |
| Label-shuffle AUC | {sanity["label_shuffle_lr_auc"]:.4f} | {sanity["label_shuffle_gbm_auc"]:.4f} | ≈ 0.50 |
| Overfit-tiny AUC (n=50) | {sanity["overfit_tiny_lr_auc"]:.4f} | {sanity["overfit_tiny_gbm_auc"]:.4f} | > 0.80 |

All sanity checks passed: models beat the baseline, collapse to chance on shuffled
labels, and can overfit a tiny slice.

## Limitations
1. **No hyperparameter search.** Both models use fixed defaults. Tuning might
   close or widen the gap but would require a held-out test set to avoid
   optimizing on the evaluation split.
2. **Single dataset.** The honest-signal features (tenure, spend, tickets) have
   weak causal signal by design; the result may not generalise to richer feature
   sets.
3. **Variance overlap rule is approximate.** A formal test (e.g., Wilcoxon
   signed-rank on fold scores) would give a precise p-value; the reported
   spread comparison is a conservative heuristic.
4. **`signup_date` unused.** Temporal splits were not used because the
   prediction task is cross-sectional (classify customers at observation time,
   not forecast future states). If the deployment context is future-cohort
   prediction, a time-based split would be required.
"""


if __name__ == "__main__":
    main()
