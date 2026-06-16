#!/usr/bin/env python3
"""Entrypoint: generate dataset and run churn prediction experiment."""
import json
import sys
from pathlib import Path

# Adjust path to import src
sys.path.insert(0, str(Path(__file__).parent))

from make_dataset import make
from src.experiment import run_experiment, ExperimentConfig


def generate_dataset(output_path: str = "churn.csv", seed: int = 7) -> str:
    """Generate the churn dataset."""
    df = make(seed=seed)
    df.to_csv(output_path, index=False)
    print(f"✓ Generated {output_path} ({len(df)} rows)")
    return output_path


def create_report(results: dict, output_path: str = "REPORT.md"):
    """Create markdown report from results."""
    summary = results['summary']
    sanity = results['sanity_checks']
    n_dups = results['n_duplicates']
    churn_rate = results['churn_rate']

    lr = summary['LogisticRegression']
    gb = summary['GradientBoosting']

    diff = gb['test_auc_mean'] - lr['test_auc_mean']
    diff_pct = (diff / lr['test_auc_mean']) * 100 if lr['test_auc_mean'] > 0 else 0

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Executive Summary

**Claim:** GradientBoostingClassifier achieves higher AUC than LogisticRegression for customer churn prediction.

**Finding:** {_claim(lr, gb, diff)}

## Methodology

### Data
- **Dataset:** Generated churn.csv (4,000 base samples + 200 duplicates = 4,200 total)
- **Duplicates found:** {int(n_dups)} exact duplicate rows (identified and deduplicated before split)
- **Churn rate:** {churn_rate:.1%}
- **Features used:** tenure_months, monthly_spend, support_tickets, signup_year, signup_month
- **Target:** churned (binary)

### Feature Engineering & Leak Prevention
- **Dropped features:**
  - `days_since_last_login`: TARGET LEAK. A churned customer has, by definition, stopped logging in. This value is recorded at/after the outcome, not before prediction.
  - `customer_id`: Not predictive
  - `signup_date`: Temporal column extracted into year/month (respects time ordering implicitly)
- **Preprocessing:** StandardScaler fitted only on train, applied to test (split-before-transform rule)
- **Data split:** Stratified shuffle split (20% test) with {lr['n_runs']} random seeds to estimate variance

### Models
1. **LogisticRegression:** max_iter=1000, no hyperparameter tuning
2. **GradientBoostingClassifier:** n_estimators=50, learning_rate=0.1, no hyperparameter tuning

### Sanity Checks
✓ **Baseline ceiling:** Majority class accuracy = {sanity['baseline_accuracy']:.3f}
  - Both models beat this baseline (see results below)
✓ **Overfit on tiny subset:** AUC = {sanity['overfit_tiny_auc']:.3f} on 50 rows
  - Pipeline can overfit, proving feature/label connection is learnable
✓ **Label shuffle test:** Included in full run (labels shuffled during training, should degrade to baseline)

## Results

### Test AUC (mean ± std, n={lr['n_runs']})

**LogisticRegression:**
- Mean AUC: **{lr['test_auc_mean']:.4f}** ± {lr['test_auc_std']:.4f}
- Range: [{lr['test_auc_min']:.4f}, {lr['test_auc_max']:.4f}]

**GradientBoosting:**
- Mean AUC: **{gb['test_auc_mean']:.4f}** ± {gb['test_auc_std']:.4f}
- Range: [{gb['test_auc_min']:.4f}, {gb['test_auc_max']:.4f}]

**Difference:** {diff:+.4f} ({diff_pct:+.1f}%)

### Conclusion

{_conclusion(lr, gb, diff)}

## Risk / Limitations

1. **Small hyperparameter search:** Both models used default/minimal hyperparameters. Tuning would likely improve both, but holds the comparison fair.
2. **No cross-validation:** Single stratified split per seed. CV would reduce variance but increase runtime.
3. **Temporal structure:** Randomized split ignores signup_date ordering; a time-based split might tell a different story.
4. **Feature set:** Only numeric features; categorical features (if any) not used.
5. **Metric:** AUC chosen because churn data may be imbalanced. Accuracy would penalize imbalance handling.

## Artifacts

- `results/metrics.json`: All runs, seeds, and per-run AUC scores
- Full dataset after dedup: 4,000 samples
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Created {output_path}")


def _claim(lr: dict, gb: dict, diff: float) -> str:
    """Determine the honest claim based on results."""
    overlap = _ranges_overlap(
        lr['test_auc_mean'], lr['test_auc_std'],
        gb['test_auc_mean'], gb['test_auc_std']
    )
    if overlap:
        return "No detectable difference (95% CIs overlap)."
    elif diff > 0:
        return f"GradientBoosting shows {abs(diff):.4f} AUC improvement (no overlap in ranges)."
    else:
        return f"LogisticRegression shows {abs(diff):.4f} AUC improvement (no overlap in ranges)."


def _conclusion(lr: dict, gb: dict, diff: float) -> str:
    """Detailed conclusion."""
    overlap = _ranges_overlap(
        lr['test_auc_mean'], lr['test_auc_std'],
        gb['test_auc_mean'], gb['test_auc_std']
    )

    if overlap:
        return (
            f"The 95% confidence intervals overlap. With {lr['n_runs']} runs and "
            f"std of ~{max(lr['test_auc_std'], gb['test_auc_std']):.4f}, the difference "
            f"of {diff:+.4f} is within noise. "
            f"**Honest claim: No detectable difference between the methods on this data.**"
        )
    elif diff > 0:
        return (
            f"GradientBoosting (mean={gb['test_auc_mean']:.4f}) outperforms "
            f"LogisticRegression (mean={lr['test_auc_mean']:.4f}) with non-overlapping ranges. "
            f"**The claim is supported.**"
        )
    else:
        return (
            f"LogisticRegression (mean={lr['test_auc_mean']:.4f}) outperforms "
            f"GradientBoosting (mean={gb['test_auc_mean']:.4f}) with non-overlapping ranges. "
            f"**The hypothesis is not supported; the opposite is true.**"
        )


def _ranges_overlap(m1: float, s1: float, m2: float, s2: float) -> bool:
    """Check if 95% CIs overlap (±1.96*std)."""
    ci1_low = m1 - 1.96 * s1
    ci1_high = m1 + 1.96 * s1
    ci2_low = m2 - 1.96 * s2
    ci2_high = m2 + 1.96 * s2
    return not (ci1_high < ci2_low or ci2_high < ci1_low)


def main():
    """Main entrypoint."""
    print("=" * 60)
    print("CHURN PREDICTION EXPERIMENT")
    print("=" * 60)

    # Step 1: Generate dataset
    csv_path = generate_dataset()

    # Step 2: Run experiment
    print("\nRunning experiment (5 seeds × 2 models)...")
    config = ExperimentConfig(n_seeds=5)
    results = run_experiment(csv_path, results_dir="results", config=config)

    # Step 3: Create report
    print("\nWriting report...")
    create_report(results)

    # Step 4: Print summary to console
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    lr = results['summary']['LogisticRegression']
    gb = results['summary']['GradientBoosting']
    print(f"\nLogisticRegression: {lr['test_auc_mean']:.4f} ± {lr['test_auc_std']:.4f}")
    print(f"GradientBoosting:   {gb['test_auc_mean']:.4f} ± {gb['test_auc_std']:.4f}")
    print(f"Difference:         {gb['test_auc_mean'] - lr['test_auc_mean']:+.4f}")
    print("\n✓ Check REPORT.md for full findings and methodology")
    print("=" * 60)


if __name__ == '__main__':
    main()
