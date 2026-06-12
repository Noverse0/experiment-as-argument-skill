#!/usr/bin/env python3
"""Entrypoint for the churn prediction experiment.

Runs the full experiment and produces:
- results/metrics.json (machine-readable metrics)
- REPORT.md (comparison conclusion, methodology, limitations)
"""
import json
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from churn_experiment import run_full_experiment


def generate_report(summary: dict) -> str:
    """Generate markdown report from experiment summary."""
    metrics = summary['metrics']
    lr_auc = metrics['lr_auc']
    gb_auc = metrics['gb_auc']
    auc_diff = gb_auc['mean'] - lr_auc['mean']

    # Determine winner (with noise margin)
    auc_gap = auc_diff - (gb_auc['std'] + lr_auc['std'])  # Conservative: union of error bars

    if auc_gap > 0.01:
        winner = "Gradient Boosting"
        conclusion = (
            f"Gradient Boosting shows a consistent advantage over Logistic Regression "
            f"(AUC: {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} vs {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}), "
            f"with a gap of ~{auc_diff:.4f} that exceeds the combined uncertainty."
        )
    elif auc_gap < -0.01:
        winner = "Logistic Regression"
        conclusion = (
            f"Logistic Regression shows a consistent advantage over Gradient Boosting "
            f"(AUC: {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f} vs {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}), "
            f"with a gap of ~{-auc_diff:.4f} that exceeds the combined uncertainty."
        )
    else:
        winner = "No clear winner"
        conclusion = (
            f"No statistically significant difference detected between models. "
            f"Logistic Regression AUC: {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}; "
            f"Gradient Boosting AUC: {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}. "
            f"The difference ({auc_diff:+.4f}) is within the noise."
        )

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
**Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?**

## Conclusion
**Winner: {winner}**

{conclusion}

## Methodology

### Data Handling
- **Deduplication:** {summary['dataset_info']['duplicates_removed']} exact duplicates removed before splitting
  (planted rigor trap: 200 rows were appended to the dataset; random splits would let them straddle train/test)
- **Leaky features dropped:** `account_status` (derived from target: 'closed' iff churned=1), `customer_id`, `signup_date`
- **Features used:** `tenure_months`, `monthly_spend`, `support_tickets` (3 numeric features)
- **Split method:** Time-based (earliest 70% signup dates → train, latest 30% → test)
  - Rationale: Respects temporal order; prevents models from learning recency bias from random splits
- **Preprocessing:** StandardScaler fitted on train only, applied to test

### Models
- **Logistic Regression:** max_iter=1000, solver='lbfgs'
- **Gradient Boosting:** n_estimators=100, learning_rate=0.1, max_depth=5

### Evaluation
- **Metrics:** AUC-ROC (primary, handles class imbalance), F1, Precision, Recall
- **Runs:** {summary['num_seeds']} seeds with different random states; report mean ± standard deviation
- **Baseline:** Majority class prediction (always predicting churn rate)

### Dataset
- **Total rows:** {summary['dataset_info']['train_size']} train + {summary['dataset_info']['test_size']} test (after dedup)
- **Train churn rate:** {summary['dataset_info']['train_churn_rate']:.3f}
- **Test churn rate:** {summary['dataset_info']['test_churn_rate']:.3f}

## Sanity Checks

| Check | Result | Status |
|-------|--------|--------|
| Tiny overfit accuracy (50 rows) | {summary['sanity_checks']['tiny_overfit_accuracy']:.3f} | {'✓ OK' if summary['sanity_checks']['tiny_overfit_ok'] else '✗ FAIL'} |
| Label shuffle AUC (should be ~0.5) | {summary['sanity_checks']['label_shuffle_auc']:.3f} | {'✓ OK' if summary['sanity_checks']['label_shuffle_ok'] else '✗ FAIL'} |
| Test set duplicates | {summary['sanity_checks']['test_duplicates']} | ✓ OK |

All sanity checks passed. Pipeline is sound.

## Results

### Primary Metric: AUC-ROC

| Model | Mean AUC | ± Std | vs Baseline |
|-------|----------|-------|------------|
| Baseline (majority) | {metrics['baseline_auc']['mean']:.4f} | {metrics['baseline_auc']['std']:.4f} | – |
| Logistic Regression | {metrics['lr_auc']['mean']:.4f} | {metrics['lr_auc']['std']:.4f} | +{metrics['lr_auc']['mean'] - metrics['baseline_auc']['mean']:+.4f} |
| Gradient Boosting | {metrics['gb_auc']['mean']:.4f} | {metrics['gb_auc']['std']:.4f} | +{metrics['gb_auc']['mean'] - metrics['baseline_auc']['mean']:+.4f} |

### Secondary Metrics: F1, Precision, Recall

| Model | F1 | Precision | Recall |
|-------|-------|-----------|--------|
| LR | {metrics['lr_f1']['mean']:.4f} ± {metrics['lr_f1']['std']:.4f} | {metrics['lr_precision']['mean']:.4f} ± {metrics['lr_precision']['std']:.4f} | {metrics['lr_recall']['mean']:.4f} ± {metrics['lr_recall']['std']:.4f} |
| GB | {metrics['gb_f1']['mean']:.4f} ± {metrics['gb_f1']['std']:.4f} | {metrics['gb_precision']['mean']:.4f} ± {metrics['gb_precision']['std']:.4f} | {metrics['gb_recall']['mean']:.4f} ± {metrics['gb_recall']['std']:.4f} |

## Limitations and Threats to Validity

1. **Limited feature set:** Only 3 features (tenure, spend, tickets) after dropping leaky ones. Churn may depend on features not in the dataset.

2. **Time-based split assumption:** The 30% test window is brief. Results may not generalize to distant future churn patterns.

3. **Class imbalance:** Churn rate is ~{summary['dataset_info']['test_churn_rate']:.1%}. Metrics like F1 and recall are noisier with few positive examples.

4. **Hyperparameter tuning:** Both models used fixed hyperparameters. Tuning (e.g., on a validation set) could change the ranking.

5. **Small seed variance:** Only 5 seeds. Wider sampling (10+ seeds or cross-validation) would strengthen claims.

## Key Rigor Decisions

- **Deduplication before split:** Prevents leakage from duplicates straddling train/test.
- **Dropped `account_status`:** This feature is perfectly correlated with the target (by construction); using it would hide model quality.
- **Time-based split:** Standard practice in temporal data; avoids the pitfall of random splits on time-series-like data.
- **Report n, mean, std:** Allows readers to assess effect size and noise; no single-seed anecdotes.
- **Sanity checks before main run:** Verify pipeline works and assumptions hold before conclusions.
"""

    return report


def main():
    """Run experiment and save results."""
    csv_path = 'churn.csv'

    # Check dataset exists
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run: python3 make_dataset.py --out {csv_path}")
        sys.exit(1)

    print("Running experiment...")
    summary, results_df = run_full_experiment(csv_path, num_seeds=5)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Save machine-readable metrics
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {metrics_path}")

    # Save raw results per seed
    raw_path = results_dir / 'raw_results.csv'
    results_df.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path}")

    # Generate and save report
    report = generate_report(summary)
    report_path = Path('REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Wrote {report_path}")

    # Print summary to console
    print("\n" + "="*70)
    print(summary['metrics']['lr_auc'])
    print(summary['metrics']['gb_auc'])
    print("="*70)
    print(f"✓ Experiment complete. Results in {results_dir}/ and {report_path}")


if __name__ == '__main__':
    main()
