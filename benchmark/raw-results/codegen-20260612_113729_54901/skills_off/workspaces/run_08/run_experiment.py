"""Entrypoint: compare LogisticRegression vs GradientBoosting on churn data."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data import load_clean, get_X_y
from src.models import make_lr, make_gb
from src.evaluate import cv_evaluate, label_shuffle_auc

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
N_SPLITS = 5


def main() -> None:
    print("=== Churn Prediction: Logistic Regression vs Gradient Boosting ===\n")

    # Generate data if not present.
    if not Path(DATA_PATH).exists():
        print(f"Generating {DATA_PATH}...")
        subprocess.run([sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True)

    # 1. Load and clean.
    print("Loading and cleaning data...")
    df, stats = load_clean(DATA_PATH)
    X, y = get_X_y(df)
    print(f"  Raw rows:          {stats['n_raw']}")
    print(f"  Duplicates dropped:{stats['n_dupes_dropped']}")
    print(f"  Clean rows:        {stats['n_clean']}")
    print(f"  Churn rate:        {stats['churn_rate']:.1%}")
    print(f"  Features:          {list(X.columns)}")
    print()

    # 2. Sanity checks.
    print("Running sanity checks...")
    # Majority-class AUC is 0.5 by definition.
    majority_auc = 0.5
    print(f"  Majority-class baseline AUC: {majority_auc:.3f}")

    shuffle_result = label_shuffle_auc(make_lr(), X, y, N_SPLITS)
    print(f"  Label-shuffle AUC (LR):      {shuffle_result['auc_mean']:.3f} ± {shuffle_result['auc_std']:.3f}  (expect ~0.5)")
    if shuffle_result["auc_mean"] > 0.55:
        print("ERROR: Label-shuffle AUC is suspiciously high — likely leakage. Aborting.")
        sys.exit(1)
    print("  Sanity checks passed.\n")

    # 3. Evaluate both models with temporal CV.
    print(f"Evaluating with TimeSeriesSplit (n_splits={N_SPLITS}) on temporal ordering...")
    lr_result = cv_evaluate(make_lr(), X, y, N_SPLITS)
    gb_result = cv_evaluate(make_gb(), X, y, N_SPLITS)

    print(f"  LogisticRegression:      AUC {lr_result['auc_mean']:.3f} ± {lr_result['auc_std']:.3f}  F1 {lr_result['f1_mean']:.3f} ± {lr_result['f1_std']:.3f}")
    print(f"  GradientBoosting:        AUC {gb_result['auc_mean']:.3f} ± {gb_result['auc_std']:.3f}  F1 {gb_result['f1_mean']:.3f} ± {gb_result['f1_std']:.3f}")
    print()

    # 4. Verdict: claim a winner only when gap exceeds noise (non-overlapping ±1 SD).
    gap = gb_result["auc_mean"] - lr_result["auc_mean"]
    lr_lo = lr_result["auc_mean"] - lr_result["auc_std"]
    lr_hi = lr_result["auc_mean"] + lr_result["auc_std"]
    gb_lo = gb_result["auc_mean"] - gb_result["auc_std"]
    gb_hi = gb_result["auc_mean"] + gb_result["auc_std"]
    ranges_overlap = lr_lo < gb_hi and gb_lo < lr_hi

    if abs(gap) < 0.01 or ranges_overlap:
        verdict = "no_detectable_difference"
        winner = None
    elif gap > 0:
        verdict = "gradient_boosting_wins"
        winner = "GradientBoosting"
    else:
        verdict = "logistic_regression_wins"
        winner = "LogisticRegression"

    print(f"  Gap (GB − LR):     {gap:+.3f}")
    print(f"  ±1 SD overlap:     {ranges_overlap}")
    print(f"  Verdict:           {verdict}")

    # 5. Persist results.
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "data_stats": stats,
        "methodology": {
            "cv": "TimeSeriesSplit",
            "n_splits": N_SPLITS,
            "primary_metric": "ROC-AUC",
            "split_order": "temporal (signup_date ascending)",
            "preprocessing": [
                "drop_exact_duplicates_before_split",
                "drop_account_status_leakage",
                "drop_customer_id",
                "signup_date_to_days_since_first",
                "StandardScaler_fit_on_train_fold_only",
            ],
        },
        "sanity_checks": {
            "majority_class_baseline_auc": majority_auc,
            "label_shuffle_auc_mean": shuffle_result["auc_mean"],
            "label_shuffle_auc_std": shuffle_result["auc_std"],
            "passed": True,
        },
        "models": {
            "LogisticRegression": lr_result,
            "GradientBoosting": gb_result,
        },
        "conclusion": {
            "gap_gb_minus_lr": float(gap),
            "lr_auc_range_1sd": [float(lr_lo), float(lr_hi)],
            "gb_auc_range_1sd": [float(gb_lo), float(gb_hi)],
            "ranges_overlap": bool(ranges_overlap),
            "verdict": verdict,
            "winner": winner,
        },
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics -> {metrics_path}")

    _write_report(metrics)
    print("Report  -> REPORT.md")


def _write_report(metrics: dict) -> None:
    lr = metrics["models"]["LogisticRegression"]
    gb = metrics["models"]["GradientBoosting"]
    c = metrics["conclusion"]
    s = metrics["data_stats"]
    sc = metrics["sanity_checks"]
    m = metrics["methodology"]

    verdict_sentences = {
        "no_detectable_difference": (
            "**No detectable difference.** The AUC gap between the two models "
            f"({c['gap_gb_minus_lr']:+.3f}) is within the ±1 SD noise band and the "
            "score ranges overlap. Neither model can be declared superior on this dataset "
            "at this sample size."
        ),
        "gradient_boosting_wins": (
            f"**Gradient Boosting outperforms Logistic Regression** "
            f"(AUC gap: {c['gap_gb_minus_lr']:+.3f}; non-overlapping ±1 SD bands)."
        ),
        "logistic_regression_wins": (
            f"**Logistic Regression outperforms Gradient Boosting** "
            f"(AUC gap: {c['gap_gb_minus_lr']:+.3f}; non-overlapping ±1 SD bands)."
        ),
    }[c["verdict"]]

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

| Dimension | Choice | Justification |
|-----------|--------|---------------|
| Variable | Model class (LR vs GBT) | All other hyperparameters and data held fixed |
| Split | TimeSeriesSplit, n={m['n_splits']} | signup_date is temporal; random splits would be leakage |
| Metric | ROC-AUC | Robust to the {s['churn_rate']:.0%} churn rate imbalance |
| Preprocessing | StandardScaler on train fold only | Prevents information from test fold reaching the scaler |
| Verdict rule | Winner only if gap > noise (non-overlapping ±1 SD) | Avoids false winner claims within noise |

**Data cleaning:**
- Dropped `account_status`: this column is `"closed"` iff `churned==1` — perfect target leakage.
- Deduplication before any split: removed {s['n_dupes_dropped']} exact duplicate rows ({s['n_raw']} → {s['n_clean']}).
- Dropped `customer_id` (row identifier, not predictive).
- Converted `signup_date` to `days_since_first` (days since earliest signup).

## Sanity Checks

| Check | Result | Threshold | Pass? |
|-------|--------|-----------|-------|
| Majority-class baseline AUC | {sc['majority_class_baseline_auc']:.3f} | ~0.5 | ✓ |
| Label-shuffle AUC (LR) | {sc['label_shuffle_auc_mean']:.3f} ± {sc['label_shuffle_auc_std']:.3f} | < 0.55 | ✓ |

## Results

| Model | AUC mean | AUC std | F1 mean | F1 std |
|-------|----------|---------|---------|--------|
| LogisticRegression | **{lr['auc_mean']:.3f}** | {lr['auc_std']:.3f} | {lr['f1_mean']:.3f} | {lr['f1_std']:.3f} |
| GradientBoosting | **{gb['auc_mean']:.3f}** | {gb['auc_std']:.3f} | {gb['f1_mean']:.3f} | {gb['f1_std']:.3f} |

AUC gap (GB − LR): **{c['gap_gb_minus_lr']:+.3f}**
LR ±1 SD: [{c['lr_auc_range_1sd'][0]:.3f}, {c['lr_auc_range_1sd'][1]:.3f}]
GB ±1 SD: [{c['gb_auc_range_1sd'][0]:.3f}, {c['gb_auc_range_1sd'][1]:.3f}]
Ranges overlap: {c['ranges_overlap']}

## Conclusion

{verdict_sentences}

## Limitations

1. **5 folds only.** With only 5 temporal folds, variance estimates are noisy. ≥10 folds or repeated k-fold would tighten the confidence interval.
2. **No hyperparameter tuning.** Both models use defaults with equal tuning budget (none). A full study would tune each arm on a held-out validation split, spending identical resources per arm.
3. **Single dataset, fixed seed.** The data-generating process is synthetic and deterministic. Results describe behaviour on one draw; generalization to real churn distributions is not established.
4. **Temporal ordering proxy.** `signup_date` drives the temporal split, but the churn labels were generated without explicit temporal drift beyond `tenure_months`. The split is methodologically correct but may understate performance variance on real drifting data.
5. **days_since_first as a feature.** Including signup timing as a feature is legitimate (it was available before the outcome), but it may correlate with cohort effects specific to this synthetic dataset.
"""

    with open("REPORT.md", "w") as fh:
        fh.write(report)


if __name__ == "__main__":
    main()
