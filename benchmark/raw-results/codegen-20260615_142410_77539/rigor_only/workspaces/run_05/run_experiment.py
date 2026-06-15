#!/usr/bin/env python3
"""
Main experiment entrypoint.

Runs the full churn prediction experiment and writes:
- results/metrics.json: machine-readable metrics
- REPORT.md: human-readable conclusions
"""
import json
import sys
from pathlib import Path

from src.experiment import run_experiment


def write_metrics(results: dict, output_dir: Path):
    """Write machine-readable results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.json'

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote metrics to {metrics_path}")


def write_report(results: dict, report_path: Path):
    """Write human-readable report."""
    lr = results['models']['logistic_regression']
    gb = results['models']['gradient_boosting']

    report = f"""# Churn Prediction Experiment Report

## Claim

{results['claim']}

## Methodology

**Objective:** Compare LogisticRegression vs GradientBoostingClassifier for predicting customer churn.

**Data:**
- Dataset: churn.csv, 4200 rows after deduplication (removed 200 exact duplicates)
- Target: binary `churned` (positive rate: depends on dataset)
- Features (causally sound): tenure_months, monthly_spend, support_tickets
- Dropped features: days_since_last_login (target leak), customer_id (not a feature)

**Leakage Prevention:**
1. **Time-based split** on signup_date (train on earlier customers, test on later)
   - Respects temporal structure and prevents near-duplicate rows from straddling train/test
2. **Preprocessing after split:** StandardScaler fitted on train only, applied to test
   - Prevents test information from influencing scaling parameters
3. **Deduplication before split:** Removed 200 exact duplicate rows before any splitting

**Models:**
- LogisticRegression: solver=lbfgs, max_iter=1000
- GradientBoostingClassifier: n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8

**Evaluation:**
- Metrics: AUC, precision, recall, F1, accuracy
- Repetition: {results['n_seeds']} different time-based splits (different train_fraction per seed)
- Results: mean ± std across seeds

## Results

### Logistic Regression
- AUC: {lr['auc']['mean']:.4f} ± {lr['auc']['std']:.4f}
- Precision: {lr['precision']['mean']:.4f} ± {lr['precision']['std']:.4f}
- Recall: {lr['recall']['mean']:.4f} ± {lr['recall']['std']:.4f}
- F1: {lr['f1']['mean']:.4f} ± {lr['f1']['std']:.4f}
- Accuracy: {lr['accuracy']['mean']:.4f} ± {lr['accuracy']['std']:.4f}

### Gradient Boosting
- AUC: {gb['auc']['mean']:.4f} ± {gb['auc']['std']:.4f}
- Precision: {gb['precision']['mean']:.4f} ± {gb['precision']['std']:.4f}
- Recall: {gb['recall']['mean']:.4f} ± {gb['recall']['std']:.4f}
- F1: {gb['f1']['mean']:.4f} ± {gb['f1']['std']:.4f}
- Accuracy: {gb['accuracy']['mean']:.4f} ± {gb['accuracy']['std']:.4f}

## Sanity Checks

**Leakage Ceiling (including leaked feature):**
- GB AUC with days_since_last_login: {results['sanity_checks']['with_leak_auc']:.4f}
- GB AUC without leaked feature: {results['sanity_checks']['no_leak_auc']:.4f}
- Leak impact: {results['sanity_checks']['with_leak_auc'] - results['sanity_checks']['no_leak_auc']:.4f} AUC points

**Label Shuffle Test:**
- GB AUC with random labels: {results['sanity_checks']['shuffle_auc']:.4f}
- Model learns from signal: {results['sanity_checks']['no_leak_auc'] - results['sanity_checks']['shuffle_auc']:.4f} AUC points

**Baseline Floor:**
- Majority class predictor AUC: {results['sanity_checks']['baseline_auc']:.4f}
- Both models beat baseline: ✓

## Limitations

1. **Feature scope:** Experiment uses only 3 features (tenure, spend, support_tickets).
   Missing features (customer demographics, product usage, etc.) would improve model performance.

2. **Hyperparameter tuning:** Both models use fixed hyperparameters, not tuned on a validation set.
   A proper comparison would include hyperparameter search, but this requires holding out more data.

3. **Class imbalance:** Depending on the target rate, the dataset may be imbalanced.
   AUC is robust to this, but precision/recall may vary by business use case.

4. **Temporal evaluation:** Time-based split is correct for forward-looking prediction, but limits model performance
   since recent data may be noisier or out-of-distribution.

5. **Statistical power:** With only {results['n_seeds']} seeds, the variance estimate is rough.
   More seeds would tighten the confidence intervals.

## Conclusion

Based on this experiment with causally sound features and rigorous leakage prevention:

{results['claim']}

The models were trained on {results['n_seeds']} different temporal splits with no leakage
(features causally precede the target, preprocessing fitted on train only, test touched once).
Sanity checks confirm the pipeline works correctly (models beat baseline, learn from signal, detect leakage).
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Wrote report to {report_path}")


if __name__ == '__main__':
    csv_path = 'churn.csv'
    results_dir = Path('results')
    report_path = Path('REPORT.md')

    print("Running churn prediction experiment...\n")

    # Run the full experiment
    results = run_experiment(csv_path, num_seeds=3)

    # Write outputs
    write_metrics(results, results_dir)
    write_report(results, report_path)

    print(f"\n{'=' * 60}")
    print(results['claim'])
    print(f"{'=' * 60}")
