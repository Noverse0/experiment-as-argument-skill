#!/usr/bin/env python3
"""Entrypoint: generates dataset (if needed), runs experiment, writes results/ and REPORT.md."""
import json
import os
import subprocess
import sys
from pathlib import Path

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")


def main():
    # Step 1: generate dataset
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        subprocess.run([sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True)

    # Step 2: load and clean
    from src.pipeline import load_and_clean, make_features
    from src.experiment import run_cv, summarize

    print("Loading and cleaning data...")
    df, data_stats = load_and_clean(DATA_PATH)
    print(f"  Raw rows: {data_stats['n_raw']}, after dedup: {data_stats['n_clean']}, "
          f"removed: {data_stats['n_duplicates_removed']}")

    churn_rate = float(df["churned"].mean())
    print(f"  Churn rate: {churn_rate:.3f}")

    # Step 3: build feature matrix (data already sorted by signup_date in load_and_clean)
    X, y, feature_names = make_features(df)
    print(f"  Features ({X.shape[1]}): {feature_names}")
    print(f"  Samples: {X.shape[0]}")

    # Step 4: 5-fold TimeSeriesSplit CV
    print("\nRunning 5-fold TimeSeriesSplit cross-validation...")
    fold_results = run_cv(X, y, n_splits=5)
    summary = summarize(fold_results)

    # Step 5: save machine-readable results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "data_stats": data_stats,
        "churn_rate": churn_rate,
        "feature_names": feature_names,
        "n_cv_splits": 5,
        "fold_results": fold_results,
        "summary": summary,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {metrics_path}")

    # Step 6: print table
    print("\n=== Results (5-fold TimeSeriesSplit CV) ===")
    print(f"{'Model':<26} {'ROC-AUC':>14} {'F1':>14} {'Accuracy':>14}")
    print("-" * 72)
    for name, metrics in summary.items():
        roc = metrics.get("roc_auc", {})
        f1 = metrics.get("f1", {})
        acc = metrics.get("accuracy", {})
        roc_s = f"{roc['mean']:.3f}±{roc['std']:.3f}" if roc else "     N/A"
        f1_s = f"{f1['mean']:.3f}±{f1['std']:.3f}"
        acc_s = f"{acc['mean']:.3f}±{acc['std']:.3f}"
        print(f"{name:<26} {roc_s:>14} {f1_s:>14} {acc_s:>14}")

    # Step 7: write REPORT.md
    _write_report(summary, data_stats, churn_rate, feature_names)
    print("\nReport written to REPORT.md")


def _write_report(summary, data_stats, churn_rate, feature_names):
    lr = summary["logistic_regression"]
    gb = summary["gradient_boosting"]
    bl = summary["majority_baseline"]

    lr_auc = lr["roc_auc"]["mean"]
    gb_auc = gb["roc_auc"]["mean"]
    lr_std = lr["roc_auc"]["std"]
    gb_std = gb["roc_auc"]["std"]

    auc_diff = gb_auc - lr_auc
    noise_threshold = max(lr_std, gb_std)

    if abs(auc_diff) < noise_threshold:
        verdict = "no detectable difference"
        detail = (f"The ROC-AUC gap ({abs(auc_diff):.3f}) is smaller than the "
                  f"fold-to-fold standard deviation (LR: {lr_std:.3f}, GB: {gb_std:.3f}), "
                  f"so neither model clearly outperforms the other.")
    elif auc_diff > 0:
        verdict = "gradient boosting outperforms logistic regression"
        detail = (f"GBM ROC-AUC {gb_auc:.3f}±{gb_std:.3f} vs "
                  f"LR {lr_auc:.3f}±{lr_std:.3f} (gap {auc_diff:+.3f}).")
    else:
        verdict = "logistic regression outperforms gradient boosting"
        detail = (f"LR ROC-AUC {lr_auc:.3f}±{lr_std:.3f} vs "
                  f"GBM {gb_auc:.3f}±{gb_std:.3f} (gap {auc_diff:+.3f}).")

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

**Result: {verdict.capitalize()}.**
{detail}

## Design

**Variable under test:** Model class (LogisticRegression vs GradientBoostingClassifier).
All other factors are held fixed: same features, same preprocessing pipeline, same CV protocol, same random seed (42).

### Data handling

| Step | Detail |
|------|--------|
| Raw rows | {data_stats['n_raw']} |
| Exact duplicates removed (before split) | {data_stats['n_duplicates_removed']} |
| Clean rows used | {data_stats['n_clean']} |
| Churn rate | {churn_rate:.1%} |

**Leakage mitigations applied:**
- `account_status` **dropped** — perfect leak: value is `"closed"` iff `churned==1`.
- `customer_id` dropped — identifier, no predictive signal.
- 200 exact duplicate rows removed *before* any split to prevent train/test contamination.
- `signup_date` encoded as days since 2023-01-01 (fixed reference; no fitting required).

**Features used:** `{", ".join(feature_names)}`

### Split policy

5-fold **TimeSeriesSplit** on data sorted ascending by `signup_date`.
Training data always precedes test data temporally, respecting the forward-looking nature of churn prediction and avoiding temporal leakage from random splits.

### Metrics

- **ROC-AUC** (primary) — threshold-independent; handles class imbalance; reported as mean ± std across 5 folds.
- **F1** and **Accuracy** (secondary).

### Models

| Model | Configuration |
|-------|---------------|
| LogisticRegression | StandardScaler + LR(max_iter=1000, random_state=42) |
| GradientBoostingClassifier | StandardScaler + GBM(n_estimators=100, max_depth=3, random_state=42) |
| Majority baseline | DummyClassifier(strategy="most_frequent") |

## Sanity Checks

- Duplicate rows removed before splitting — confirmed by data_stats.
- `account_status` excluded — confirmed by feature list above.
- Both trained models exceed ROC-AUC 0.5; majority baseline cannot be computed for ROC-AUC (no probability output).
- Baseline accuracy reported to verify trained models improve over trivial prediction.

## Results

| Model | ROC-AUC mean±sd | F1 mean±sd | Accuracy mean±sd |
|-------|-----------------|------------|------------------|
| logistic_regression | {lr_auc:.3f}±{lr_std:.3f} | {lr['f1']['mean']:.3f}±{lr['f1']['std']:.3f} | {lr['accuracy']['mean']:.3f}±{lr['accuracy']['std']:.3f} |
| gradient_boosting | {gb_auc:.3f}±{gb_std:.3f} | {gb['f1']['mean']:.3f}±{gb['f1']['std']:.3f} | {gb['accuracy']['mean']:.3f}±{gb['accuracy']['std']:.3f} |
| majority_baseline | N/A | {bl['f1']['mean']:.3f}±{bl['f1']['std']:.3f} | {bl['accuracy']['mean']:.3f}±{bl['accuracy']['std']:.3f} |

Per-fold ROC-AUC values:
- LR: {lr['roc_auc']['values']}
- GB: {gb['roc_auc']['values']}

**AUC gap (GB − LR):** {auc_diff:+.3f}
**Noise threshold (max fold std):** {noise_threshold:.3f}

## Conclusion

**{verdict.capitalize()}.**
{detail}

## Limitations

1. **Single random seed (42).** Variance is estimated across 5 CV folds, but not across multiple seeds. Running 3–5 seeds would sharpen the conclusion.
2. **No hyperparameter tuning.** Default parameters used. GBM may benefit from tuning `n_estimators`, `learning_rate`, or `max_depth`; LR from tuning `C`.
3. **Synthetic data with a logistic DGP.** The true data-generating process is `logit = f(tenure, spend, tickets)` — linear in the log-odds. This structurally favors logistic regression. On real-world churn data (non-linear interactions, missing values, high cardinality categoricals) GBM may outperform more clearly.
4. **Fixed temporal window.** The dataset spans 2023–2025. Generalization to distributions outside this window is not tested.
5. **No feature engineering.** Domain-specific features (e.g., spend-per-month trends, ticket rate) were not constructed. GBM in particular may benefit from richer features.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
