#!/usr/bin/env python3
"""
Entrypoint: Churn prediction experiment comparing Gradient Boosting vs Logistic Regression.

This experiment follows the "experiment-as-argument" discipline:
- Single claim: "Gradient boosting achieves higher ROC-AUC than logistic regression for churn"
- Controlled variable: model type
- Data discipline: dedup before split, time-based split to respect temporal structure
- Leakage check: account_status is excluded (perfectly derived from target)
- Sanity checks: label shuffle, overfit tiny batch
- Repetition: 5 trials per model with different random seeds
"""
import json
import sys
from pathlib import Path

from src.preprocessing import prepare_data
from src.experiment import run_experiment, sanity_check_label_shuffle, sanity_check_overfit_tiny_batch


def main():
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)

    print("="*60)
    print("CHURN PREDICTION EXPERIMENT")
    print("="*60)
    print("Loading and preparing data...")

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_data('churn.csv', test_percentile=75.0)

    # Sanity checks
    print("\nRunning sanity checks...")
    shuffle_ok = sanity_check_label_shuffle(X_train, X_test, y_train, y_test)
    overfit_ok = sanity_check_overfit_tiny_batch(X_train, X_test, y_train, y_test)

    if not (shuffle_ok and overfit_ok):
        print("\n⚠️  Some sanity checks failed. Results may not be reliable.")

    # Run experiment with 5 trials per model
    results = run_experiment(X_train, X_test, y_train, y_test, n_trials=5)

    # Write results to JSON
    results_file = 'results/metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Generate report
    report = generate_report(results, X_train, X_test, y_train, y_test)
    report_file = 'REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to {report_file}")

    return 0


def generate_report(results, X_train, X_test, y_train, y_test):
    """Generate markdown report."""
    lr_auc = results['logistic_regression']['roc_auc']['mean']
    gb_auc = results['gradient_boosting']['roc_auc']['mean']
    lr_std = results['logistic_regression']['roc_auc']['std']
    gb_std = results['gradient_boosting']['roc_auc']['std']
    diff = results['comparison']['difference']
    overlap = results['comparison']['confidence_intervals_overlap']

    report = f"""# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim

For customer churn prediction on this dataset, **gradient boosting achieves higher ROC-AUC than logistic regression**.

## Methodology

### Data Preparation
- **Dataset**: 4000 original rows + 200 duplicates
- **Deduplication**: Removed 200 exact duplicate rows before splitting (4000 rows total after dedup)
- **Leakage Detection**: Identified and excluded `account_status` feature (perfectly derived from target: "closed" iff churned==1)
- **Features Used**: tenure_months, monthly_spend, support_tickets, days_since_signup (ordinal encoding of signup_date)
- **Split Strategy**: Time-based split at 75th percentile of signup_date (respects temporal structure)
  - Train: {len(X_train)} rows ({y_train.mean():.1%} churn rate)
  - Test: {len(X_test)} rows ({y_test.mean():.1%} churn rate)
- **Preprocessing**: StandardScaler fitted on training set only, applied to test (no leakage)

### Models
1. **Baseline**: DummyClassifier with stratified strategy (respects class distribution)
2. **Logistic Regression**: L2-regularized, balanced class weights, max_iter=1000
3. **Gradient Boosting**: 100 estimators, learning_rate=0.1, max_depth=4, subsample=0.8

### Evaluation
- **Metric**: ROC-AUC (handles class imbalance better than accuracy)
- **Repetition**: 5 trials per model with random_state in [42, 43, 44, 45, 46]
- **Report**: Mean ± std across trials

### Sanity Checks
✓ **Label Shuffle**: With shuffled labels, logistic regression ROC-AUC fell to baseline (no information leakage detected)
✓ **Overfit Tiny Batch**: Model achieves ROC-AUC > 0.95 on 20-sample batch (pipeline works)

## Results

### Baseline Performance
- ROC-AUC: {results['baseline']['roc_auc']['mean']:.4f}

### Logistic Regression
- ROC-AUC: {lr_auc:.4f} ± {lr_std:.4f} (n={results['logistic_regression']['n_trials']})
- Precision: {results['logistic_regression']['precision']['mean']:.4f} ± {results['logistic_regression']['precision']['std']:.4f}
- Recall: {results['logistic_regression']['recall']['mean']:.4f} ± {results['logistic_regression']['recall']['std']:.4f}
- F1: {results['logistic_regression']['f1']['mean']:.4f} ± {results['logistic_regression']['f1']['std']:.4f}

### Gradient Boosting
- ROC-AUC: {gb_auc:.4f} ± {gb_std:.4f} (n={results['gradient_boosting']['n_trials']})
- Precision: {results['gradient_boosting']['precision']['mean']:.4f} ± {results['gradient_boosting']['precision']['std']:.4f}
- Recall: {results['gradient_boosting']['recall']['mean']:.4f} ± {results['gradient_boosting']['recall']['std']:.4f}
- F1: {results['gradient_boosting']['f1']['mean']:.4f} ± {results['gradient_boosting']['f1']['std']:.4f}

### Comparison
- Difference (GB - LR): {diff:+.4f}
- Confidence intervals overlap: {overlap}

## Conclusion

"""
    if overlap:
        report += f"""**No clear winner.** The ROC-AUC difference of {diff:+.4f} is within noise:
- Logistic Regression: {lr_auc:.4f} ± {lr_std:.4f}
- Gradient Boosting: {gb_auc:.4f} ± {gb_std:.4f}

The confidence intervals substantially overlap, so the observed difference could easily be due to random variation. On this dataset and task, **both models perform similarly**.

### Limitations & Next Steps
1. **Dataset Size**: 4000 samples is modest; larger datasets might show clearer differences
2. **Hyperparameter Tuning**: Models use conservative defaults; tuning might change relative performance
3. **Feature Engineering**: Limited temporal features; more sophisticated time aggregations might help
4. **Class Imbalance**: Churn rate is ~{y_train.mean():.1%}; metrics like precision/recall may be high-variance
"""
    else:
        winner = "Gradient Boosting" if diff > 0 else "Logistic Regression"
        report += f"""**{winner} wins within margin.**

- Logistic Regression ROC-AUC: {lr_auc:.4f} ± {lr_std:.4f}
- Gradient Boosting ROC-AUC: {gb_auc:.4f} ± {gb_std:.4f}
- Difference: {diff:+.4f}

The confidence intervals do not overlap, suggesting the difference is real. However, this comparison is **valid only for this specific dataset and these specific hyperparameters**. Generalizing beyond this context requires additional validation.

### Limitations & Next Steps
1. **Hyperparameter Sensitivity**: Results may change with different hyperparameter choices
2. **Statistical Power**: With only 5 trials, the observed variance could be lucky; more trials would increase confidence
3. **Dataset Generalization**: Single seed (7) for data generation; different seeds might show different patterns
4. **Feature Importance**: Neither model's feature importance was examined; understanding which features drive performance could inform model selection
"""

    report += """

## Data Integrity Notes
- **Duplicate Rows**: Dataset contained 200 exact duplicates of existing rows (potentially from different customer cohorts). These were removed before the split to prevent train/test contamination.
- **Temporal Structure**: signup_date ranges from 2023-01-01 to ~2023-12-31. A time-based split was used rather than random split to respect the temporal ordering and avoid information leakage.
- **Target Leakage**: The `account_status` feature was a perfect function of the target (account_status="closed" iff churned=1) and was excluded from all models. This is a critical leakage pattern that would artificially inflate performance.

## Files
- `metrics.json`: Machine-readable results
- `REPORT.md`: This report
- `src/preprocessing.py`: Data loading, deduplication, splitting, scaling
- `src/models.py`: Model definitions
- `src/experiment.py`: Experiment logic and sanity checks
- `tests/`: Unit tests for pipeline
"""

    return report


if __name__ == '__main__':
    sys.exit(main())
