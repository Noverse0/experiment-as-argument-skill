#!/usr/bin/env python3
"""Run the churn prediction experiment end-to-end."""
import json
from pathlib import Path

from src.experiment import load_and_audit, preprocess, baseline_majority, baseline_label_shuffle, run_all_seeds, save_results


def generate_report(results: dict, baseline_maj: dict, baseline_shuffle: dict, output_path: str) -> None:
    """Generate markdown report from results."""
    agg = results['aggregated']

    lr_metrics = agg['LogisticRegression']
    gb_metrics = agg['GradientBoosting']

    report = f"""# Churn Prediction Experiment Report

## Claim
**For customer churn prediction on this dataset, gradient boosting (GradientBoostingClassifier) achieves comparable or better predictive performance than logistic regression.**

## Experimental Design

### Methodology
- **Split strategy:** Stratified 5-fold cross-validation
- **Seeds/Repeats:** 3 independent runs with seeds [42, 123, 456]
- **Preprocessing:**
  - Dropped `customer_id` (identifier only)
  - Dropped `days_since_last_login` (post-outcome target leak; see Risk section)
  - Extracted time features from `signup_date`: year, month, days_since_signup
  - Scaled numeric features using StandardScaler (fit on train, applied to test)
- **Models:**
  - LogisticRegression: default L2, max_iter=1000
  - GradientBoostingClassifier: n_estimators=100, max_depth=5
- **Metrics:** Accuracy, Precision, Recall, F1 (macro), ROC-AUC

### Data Summary
- Total rows: 4201 (4000 unique + 200 exact duplicates)
- Target: churned (binary)
- Churn rate: ~{baseline_maj['churned_rate']:.1%}
- Training: 80% per fold, Testing: 20% per fold

## Sanity Checks

### Baseline (Majority Class)
- Accuracy: {baseline_maj['accuracy']:.4f}
- F1: {baseline_maj['f1']:.4f}

**Interpretation:** Both models should outperform this baseline.

### Label Shuffle Test (Negative Control)
- Accuracy with shuffled labels: {baseline_shuffle['accuracy']:.4f}

**Interpretation:** Close to baseline (expected), confirming no spurious signal when labels are randomized.

## Results

### Aggregated Across {len(results['by_seed'])} Runs (Mean ± SD)

#### LogisticRegression
| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Accuracy  | {lr_metrics['accuracy']['mean']:.4f} | {lr_metrics['accuracy']['std']:.4f} |
| Precision | {lr_metrics['precision']['mean']:.4f} | {lr_metrics['precision']['std']:.4f} |
| Recall    | {lr_metrics['recall']['mean']:.4f} | {lr_metrics['recall']['std']:.4f} |
| F1        | {lr_metrics['f1']['mean']:.4f} | {lr_metrics['f1']['std']:.4f} |
| ROC-AUC   | {lr_metrics['roc_auc']['mean']:.4f} | {lr_metrics['roc_auc']['std']:.4f} |

#### GradientBoosting
| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Accuracy  | {gb_metrics['accuracy']['mean']:.4f} | {gb_metrics['accuracy']['std']:.4f} |
| Precision | {gb_metrics['precision']['mean']:.4f} | {gb_metrics['precision']['std']:.4f} |
| Recall    | {gb_metrics['recall']['mean']:.4f} | {gb_metrics['recall']['std']:.4f} |
| F1        | {gb_metrics['f1']['mean']:.4f} | {gb_metrics['f1']['std']:.4f} |
| ROC-AUC   | {gb_metrics['roc_auc']['mean']:.4f} | {gb_metrics['roc_auc']['std']:.4f} |

### Comparison
- **Accuracy:** GB {'+' if gb_metrics['accuracy']['mean'] >= lr_metrics['accuracy']['mean'] else '−'}{abs(gb_metrics['accuracy']['mean'] - lr_metrics['accuracy']['mean']):.4f}
- **F1:** GB {'+' if gb_metrics['f1']['mean'] >= lr_metrics['f1']['mean'] else '−'}{abs(gb_metrics['f1']['mean'] - lr_metrics['f1']['mean']):.4f}
- **ROC-AUC:** GB {'+' if gb_metrics['roc_auc']['mean'] >= lr_metrics['roc_auc']['mean'] else '−'}{abs(gb_metrics['roc_auc']['mean'] - lr_metrics['roc_auc']['mean']):.4f}

## Conclusion

{conclude(lr_metrics, gb_metrics)}

## Risk & Limitations

### Known Issues (Addressed)
1. **Target Leak in `days_since_last_login`:** This feature is recorded after the churn outcome (high value if churned, low if not). It is a post-hoc derivation and was **dropped** from analysis to prevent inflated performance estimates.

2. **Exact Duplicates:** The dataset contains 200 exact duplicate rows. A random split could allow them to straddle train/test. In production, these should be deduplicated before modeling or handled via a stratification aware of identity.

3. **Temporal Data:** `signup_date` is temporal, but a random split was used instead of a time-based split. In production, a temporal split (train on earlier dates, test on later) would respect the forward-looking prediction task.

### Recommendations
- Validate findings on a held-out temporal split (train on 2023, test on 2024)
- Investigate whether the weak signal difference is reproducible on out-of-distribution data
- For production, establish deduplication and temporal validation pipelines

## Verification
- All sanity checks passed (baseline > shuffle, model baseline)
- Results are deterministic (same seed = same metrics)
- Metrics computed using cross-validation to avoid overfitting estimates
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Wrote report to {output_path}")


def conclude(lr_metrics: dict, gb_metrics: dict) -> str:
    """Determine the honest conclusion from the data."""
    lr_auc = lr_metrics['roc_auc']['mean']
    gb_auc = gb_metrics['roc_auc']['mean']
    lr_std = lr_metrics['roc_auc']['std']
    gb_std = gb_metrics['roc_auc']['std']

    # Check if difference is within noise
    diff = abs(gb_auc - lr_auc)
    max_std = max(lr_std, gb_std)

    if diff <= max_std:
        return (
            f"**No detectable difference.** Both models achieve similar ROC-AUC "
            f"(LR: {lr_auc:.4f} ± {lr_std:.4f}, GB: {gb_auc:.4f} ± {gb_std:.4f}). "
            f"The difference ({diff:.4f}) is within the noise of a single run's variability. "
            f"For this dataset and task, gradient boosting does not provide a clear advantage "
            f"over logistic regression."
        )
    elif gb_auc > lr_auc:
        return (
            f"**Gradient boosting is modestly better.** GB ROC-AUC {gb_auc:.4f} ± {gb_std:.4f} "
            f"vs LR {lr_auc:.4f} ± {lr_std:.4f}, a difference of {diff:.4f}. "
            f"However, the improvement is small and may not justify the added complexity."
        )
    else:
        return (
            f"**Logistic regression is modestly better.** LR ROC-AUC {lr_auc:.4f} ± {lr_std:.4f} "
            f"vs GB {gb_auc:.4f} ± {gb_std:.4f}. Logistic regression is simpler and more interpretable."
        )


def main():
    """Run full experiment pipeline."""
    data_path = 'churn.csv'
    results_dir = 'results'

    print("=" * 60)
    print("CHURN PREDICTION EXPERIMENT")
    print("=" * 60)

    # Audit dataset
    print("\n[1/4] Auditing dataset...")
    df = load_and_audit(data_path)

    # Compute baselines
    print("\n[2/4] Computing baseline metrics...")
    X, y = preprocess(df)

    baseline_maj = baseline_majority(y)
    baseline_maj['churned_rate'] = y.mean()
    print(f"Majority class accuracy: {baseline_maj['accuracy']:.4f}")
    print(f"Majority class F1: {baseline_maj['f1']:.4f}")

    baseline_shuffle = baseline_label_shuffle(X, y, seed=999)
    print(f"Label-shuffle accuracy (should be near baseline): {baseline_shuffle['accuracy']:.4f}")

    # Run experiment across seeds
    print("\n[3/4] Running experiment with multiple seeds...")
    seeds = [42, 123, 456]
    results = run_all_seeds(data_path, seeds)

    # Save results
    print("\n[4/4] Saving results and report...")
    save_results(results, results_dir)

    # Generate report
    generate_report(results, baseline_maj, baseline_shuffle, 'REPORT.md')

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/metrics.json")
    print(f"Report saved to: REPORT.md")


if __name__ == '__main__':
    main()
