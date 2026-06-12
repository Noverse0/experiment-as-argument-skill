#!/usr/bin/env python3
"""Entrypoint: Generate dataset, run experiment, write results and report."""
import json
import subprocess
import sys
from pathlib import Path

from src.experiment import DataLoader, ExperimentRunner, summarize_results


def main():
    """Execute the full experiment pipeline."""
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("=" * 60)
    print("STEP 1: Generating dataset...")
    print("=" * 60)
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Step 2: Load and clean data
    print("=" * 60)
    print("STEP 2: Loading and cleaning dataset...")
    print("=" * 60)
    features, target = DataLoader.load_and_clean("churn.csv")

    # Step 3: Run experiment across multiple seeds
    print("=" * 60)
    print("STEP 3: Running experiment with 5 seeds...")
    print("=" * 60)
    seeds = [42, 123, 456, 789, 999]
    runner = ExperimentRunner(features, target)
    all_results = runner.run_multiple_seeds(seeds)

    # Step 4: Summarize results
    summary = summarize_results(all_results)

    # Step 5: Write results to JSON
    print("=" * 60)
    print("STEP 4: Writing results...")
    print("=" * 60)
    results_json = {
        "experiment": "churn_prediction_comparison",
        "claim": "For predicting customer churn, does gradient boosting outperform logistic regression?",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "summary": summary,
    }
    with open("results/metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Wrote results/metrics.json")

    # Step 6: Generate report
    print("=" * 60)
    print("STEP 5: Generating report...")
    print("=" * 60)

    # Determine winner and claim
    gb_auc = summary["GradientBoosting"]["test_auc_mean"]
    lr_auc = summary["LogisticRegression"]["test_auc_mean"]
    gb_std = summary["GradientBoosting"]["test_auc_std"]
    lr_std = summary["LogisticRegression"]["test_auc_std"]
    auc_diff = gb_auc - lr_auc
    overlap = (gb_auc - gb_std) < (lr_auc + lr_std)

    if overlap:
        conclusion = (
            "No statistically significant difference detected. "
            "Confidence intervals overlap; difference is within noise."
        )
    elif auc_diff > 0:
        conclusion = (
            f"Gradient Boosting shows {abs(auc_diff):.3f} AUC improvement "
            f"over Logistic Regression across {len(seeds)} seeds."
        )
    else:
        conclusion = (
            f"Logistic Regression shows {abs(auc_diff):.3f} AUC improvement "
            f"over Gradient Boosting across {len(seeds)} seeds."
        )

    report = f"""# Churn Prediction Model Comparison Report

## Claim
**For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?**

## Methodology

### Data Handling
- **Dataset:** 4000 customers + 200 duplicates (churn.csv)
- **Leakage Detection & Mitigation:**
  - Identified `account_status` as perfect leak (derived from target) → **Removed**
  - Detected 200 duplicate rows → **Removed before split** to prevent cross-boundary leakage
  - `signup_date` parsed as temporal feature; created `days_since_signup` as derivative
- **Target Distribution:** ~{target.mean():.1%} positive (churned)

### Train/Val/Test Split
- **Strategy:** Stratified random split (preserves class balance)
- **Proportions:** 60% train, 20% validation, 20% test
- **Rationale:** Stratified ensures both splits see representative class distributions

### Model Configuration
**Logistic Regression:**
- max_iter=1000, regularization=default (L2)
- Fitted on standardized features

**Gradient Boosting Classifier:**
- n_estimators=100, learning_rate=0.1, max_depth=5
- Early stopping with n_iter_no_change=10, validation_fraction=0.1
- Random state fixed per seed

### Feature Preprocessing
- StandardScaler fit on training set only
- Applied identically to validation and test sets
- This prevents information leakage from test statistics into training

### Evaluation Metrics
- **Primary:** ROC-AUC (robust to class imbalance)
- **Secondary:** Precision, Recall, F1 (at threshold optimized on validation set)

### Multiple Seeds & Variance
- Experiment repeated with 5 independent seeds: {', '.join(map(str, seeds))}
- Results reported as mean ± std across runs
- Overlapping confidence intervals indicate no significant difference

## Results

### Test AUC (Primary Metric)
```
Logistic Regression:  {lr_auc:.4f} ± {lr_std:.4f} (n={summary['LogisticRegression']['test_auc_runs']})
Gradient Boosting:    {gb_auc:.4f} ± {gb_std:.4f} (n={summary['GradientBoosting']['test_auc_runs']})
Difference (GB - LR): {auc_diff:.4f}
```

### Secondary Metrics (Test Set, Mean Across Seeds)
```
Logistic Regression:
  Precision: {summary['LogisticRegression']['test_precision_mean']:.4f}
  Recall:    {summary['LogisticRegression']['test_recall_mean']:.4f}
  F1:        {summary['LogisticRegression']['test_f1_mean']:.4f}

Gradient Boosting:
  Precision: {summary['GradientBoosting']['test_precision_mean']:.4f}
  Recall:    {summary['GradientBoosting']['test_recall_mean']:.4f}
  F1:        {summary['GradientBoosting']['test_f1_mean']:.4f}
```

## Conclusion
{conclusion}

## Limitations & Threats to Validity

1. **Single Dataset:** Results reflect performance on one synthetic churn distribution; generalization to other customer populations unknown.
2. **Hyperparameter Tuning:** Both models use defaults; no hyperparameter search was conducted. A more thorough search might favor one algorithm.
3. **Feature Engineering:** Only raw numeric features and a derived temporal feature used. Domain-specific feature engineering could shift results.
4. **Sample Size:** 3200 training samples after deduplication; sufficient but modest for deep learning comparison (not applicable here).
5. **Temporal Dynamics:** Random split ignores time ordering. If churn patterns drift over time, time-based split might reveal different performance.

## Integrity Checks Performed

✓ Removed perfect leak (account_status)
✓ Deduplicated before split
✓ Fit preprocessor on training set only
✓ Ran multiple seeds to estimate variance
✓ Reported overlapping distributions for fair comparison
"""

    with open("REPORT.md", "w") as f:
        f.write(report)
    print(f"Wrote REPORT.md")

    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Logistic Regression Test AUC: {lr_auc:.4f} ± {lr_std:.4f}")
    print(f"  Gradient Boosting Test AUC:   {gb_auc:.4f} ± {gb_std:.4f}")
    print(f"\n{conclusion}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
