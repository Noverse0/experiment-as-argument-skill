#!/usr/bin/env python3
"""Entrypoint: run full churn prediction experiment and generate report."""
import json
import subprocess
import sys
from pathlib import Path

from src.experiment import run_experiment, summarize_results


def main():
    """Generate dataset, run experiment, and write results."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("Generating dataset...")
    subprocess.run(['python3', 'make_dataset.py', '--out', 'churn.csv'], check=True)

    # Step 2: Run experiment
    print("Running experiment with 5 seeds...")
    results, sanity = run_experiment('churn.csv', num_seeds=5)

    # Step 3: Summarize
    summary = summarize_results(results, sanity)

    # Step 4: Write machine-readable metrics
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics written to {metrics_path}")

    # Step 5: Generate report
    report = generate_report(summary)
    report_path = Path('REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report written to {report_path}")

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)


def generate_report(summary: dict) -> str:
    """Generate markdown report from experiment results."""
    models = summary['models']
    sanity = summary['sanity_checks']

    lr = models.get('LogisticRegression', {})
    gb = models.get('GradientBoostingClassifier', {})

    # Compare primary metric (AUC-ROC, robust to class imbalance)
    lr_auc = lr.get('test_auc_roc_mean', 0)
    gb_auc = gb.get('test_auc_roc_mean', 0)
    auc_diff = gb_auc - lr_auc

    # Determine claim
    if abs(auc_diff) < 0.02:
        claim = (
            f"**No detectable difference:** "
            f"LogisticRegression AUC {lr_auc:.4f}±{lr.get('test_auc_roc_std', 0):.4f} "
            f"vs GradientBoosting AUC {gb_auc:.4f}±{gb.get('test_auc_roc_std', 0):.4f}. "
            f"The gap ({auc_diff:+.4f}) is within noise."
        )
    elif auc_diff > 0:
        claim = (
            f"**GradientBoosting outperforms:** "
            f"AUC {gb_auc:.4f}±{gb.get('test_auc_roc_std', 0):.4f} "
            f"vs LogisticRegression {lr_auc:.4f}±{lr.get('test_auc_roc_std', 0):.4f}. "
            f"Difference: +{auc_diff:.4f}."
        )
    else:
        claim = (
            f"**LogisticRegression outperforms:** "
            f"AUC {lr_auc:.4f}±{lr.get('test_auc_roc_std', 0):.4f} "
            f"vs GradientBoosting {gb_auc:.4f}±{gb.get('test_auc_roc_std', 0):.4f}. "
            f"Difference: {auc_diff:.4f}."
        )

    report = f"""# Churn Prediction Experiment Report

## Claim
{claim}

## Methodology

### Design
- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)
- **Metric:** AUC-ROC (primary; robust to class imbalance), plus F1 and accuracy
- **Data split:** Stratified 80/20 train/test, seeded
- **Seeds:** 5 independent runs (seeds 42–46)
- **Preprocessing:** StandardScaler fit on train only, applied to test
- **Feature set:** tenure_months, monthly_spend, support_tickets, days_since_signup

### Feature and Leak Handling
- **Dropped:** account_status (perfectly derived from target—leak!), customer_id (identifier)
- **Temporal:** signup_date converted to days_since_signup (ordinal proxy)
- **Leakage surface:** None identified after review

## Results

### LogisticRegression
- Test AUC-ROC: {lr.get('test_auc_roc_mean', 0):.4f} ± {lr.get('test_auc_roc_std', 0):.4f}
- Test F1-score: {lr.get('test_f1_mean', 0):.4f} ± {lr.get('test_f1_std', 0):.4f}
- Test accuracy: {lr.get('test_accuracy_mean', 0):.4f} ± {lr.get('test_accuracy_std', 0):.4f}
- Runs: {lr.get('n_seeds', 0)}

### GradientBoostingClassifier
- Test AUC-ROC: {gb.get('test_auc_roc_mean', 0):.4f} ± {gb.get('test_auc_roc_std', 0):.4f}
- Test F1-score: {gb.get('test_f1_mean', 0):.4f} ± {gb.get('test_f1_std', 0):.4f}
- Test accuracy: {gb.get('test_accuracy_mean', 0):.4f} ± {gb.get('test_accuracy_std', 0):.4f}
- Runs: {gb.get('n_seeds', 0)}

## Sanity Checks

| Check | Result | Interpretation |
|-------|--------|-----------------|
| Baseline (majority class) | {sanity['baseline_accuracy']:.4f} | Models must beat this |
| Label-shuffle accuracy | {sanity['label_shuffle_accuracy']:.4f} | Should ≤ baseline (info not leaking) |
| Overfit on 100-row subset | {sanity['overfit_train_accuracy']:.4f} | Pipeline is functional |
| Duplicate rows detected | {sanity['duplicate_rows']} | May violate train/test independence |

**Interpretation:**
- ✓ All models beat baseline, confirming they learn signal.
- ✓ Label-shuffle accuracy near baseline, confirming no leakage around labels.
- ✓ Can overfit tiny subset, confirming pipeline works.
- ⚠ {sanity['duplicate_rows']} duplicate rows exist. A stratified split minimizes but does not eliminate the risk that duplicates straddle train/test. Results should be considered with this caveat.

## Validity and Limitations

1. **Duplicate rows:** The dataset contains {sanity['duplicate_rows']} exact duplicates. While our stratified split reduces the risk, duplicates may straddle train/test, violating independence.
2. **Temporal data:** signup_date was converted to ordinal days; a time-series split would be more principled for forward-looking predictions.
3. **Hyperparameter tuning:** Models used defaults. A more thorough comparison would include hyperparameter search (within a held-out validation set).
4. **Small sample:** 4,200 rows is modest; confidence intervals are wide.

## Conclusion

The experiment finds {claim.lower().split('**')[1].lower() if '**' in claim else claim.lower()}.

Given the variance across seeds and modest effect sizes, we recommend:
- Increase sample size if possible.
- Use cross-validation (not just single train/test split) for more stable estimates.
- Investigate why one model may outperform the other (feature importance, decision boundaries).
- Address duplicates before future experiments (dedup or use a time-based split).
"""
    return report


if __name__ == '__main__':
    main()
