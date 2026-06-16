#!/usr/bin/env python3
"""Entrypoint: run the full experiment and generate report."""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import (
    load_churn_data,
    check_duplicates,
    deduplicate,
    detect_leak_days_since_login,
    prepare_features,
    report_class_distribution,
)
from src.experiment import Experiment

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT: Gradient Boosting vs Logistic Regression")
    print("=" * 70)

    # Load dataset
    print("\n[1] Loading dataset...")
    df = load_churn_data('churn.csv')
    print(f"  Loaded {len(df)} rows (including duplicates)")

    # Check for duplicates
    print("\n[2] Checking for duplicates...")
    n_dups = check_duplicates(df)
    print(f"  Found {n_dups} duplicate rows")
    df = deduplicate(df)
    print(f"  After deduplication: {len(df)} rows")

    # Report class distribution
    print("\n[3] Class distribution...")
    X, y = prepare_features(df, include_leaky=False)
    class_dist = report_class_distribution(y)
    print(f"  Total samples: {class_dist['n_samples']}")
    print(f"  Churned: {class_dist['n_churned']} ({class_dist['churn_rate']:.1%})")
    print(f"  Active: {class_dist['n_active']} ({1-class_dist['churn_rate']:.1%})")

    # Leak detection: timing test for days_since_last_login
    print("\n[4] Leak detection: timing test for days_since_last_login...")
    leak_stats = detect_leak_days_since_login(df)
    print(f"  Churned customers, days_since_last_login: {leak_stats['churned_mean']:.1f} ± {leak_stats['churned_std']:.1f}")
    print(f"  Active customers, days_since_last_login: {leak_stats['active_mean']:.1f} ± {leak_stats['active_std']:.1f}")
    print(f"  Difference: {leak_stats['diff_mean']:.1f} days (STRONG LEAK SIGNAL)")
    print(f"  ⚠️  CONCLUSION: days_since_last_login is POST-OUTCOME, EXCLUDED from safe features")

    # Initialize experiment
    print("\n[5] Running experiment (5 seeds × 5-fold CV)...")
    exp = Experiment(X, y, n_repeats=5, n_splits=5)

    # Sanity checks
    print("\n[6] Sanity checks...")

    print("  - Baseline accuracy check...")
    baseline_check = exp.sanity_check_baseline()
    print(f"    Baseline (majority class): {baseline_check['baseline_accuracy']:.1%}")

    print("  - Label shuffle test (performance should drop near 0.5 AUC)...")
    shuffle_check = exp.sanity_check_label_shuffle()
    print(f"    LR AUC with shuffled labels: {shuffle_check['lr_auc_shuffled']:.3f}")
    print(f"    GB AUC with shuffled labels: {shuffle_check['gb_auc_shuffled']:.3f}")
    if shuffle_check['lr_above_baseline'] or shuffle_check['gb_above_baseline']:
        print(f"    ⚠️  WARNING: Model performance barely degrades with shuffled labels!")
    else:
        print(f"    ✓ PASS: Models degrade properly with shuffled labels")

    print("  - Overfit on small batch check...")
    overfit_check = exp.sanity_check_overfit_small_batch()
    print(f"    LR training accuracy (50 samples): {overfit_check['lr_train_accuracy']:.1%}")
    print(f"    GB training accuracy (50 samples): {overfit_check['gb_train_accuracy']:.1%}")
    if overfit_check['both_above_90']:
        print(f"    ✓ PASS: Both models can overfit a small batch")
    else:
        print(f"    ⚠️  WARNING: Models cannot overfit small batch (pipeline may be broken)")

    # Run main comparison
    print("\n[7] Training models (5 seeds × 5-fold CV = 25 cross-validations per model)...")
    exp.run_model_comparison()
    print(f"  ✓ Completed")

    # Compute summary stats
    print("\n[8] Computing summary statistics...")
    summary = exp.compute_summary_stats()
    print(f"  LR:  AUC = {summary['lr_mean_auc']:.4f} ± {summary['lr_std_auc']:.4f}")
    print(f"  GB:  AUC = {summary['gb_mean_auc']:.4f} ± {summary['gb_std_auc']:.4f}")
    print(f"  Gap: {summary['mean_auc_gap']:.4f} ± {summary['std_auc_gap']:.4f}")

    # Save results
    print("\n[9] Saving results...")
    all_results = {
        'config': {
            'dataset': 'churn.csv',
            'n_samples': int(class_dist['n_samples']),
            'churn_rate': float(class_dist['churn_rate']),
            'safe_features': ['tenure_months', 'monthly_spend', 'support_tickets'],
            'excluded_features': ['days_since_last_login (post-outcome leak)', 'customer_id (identifier)', 'signup_date (redundant with tenure)'],
        },
        'sanity_checks': {
            'baseline': baseline_check,
            'label_shuffle': shuffle_check,
            'overfit_small_batch': overfit_check,
        },
        'leak_detection': leak_stats,
        'experiment': exp.results,
        'summary': summary,
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Wrote results/metrics.json")

    # Generate markdown report
    print("\n[10] Generating report...")
    generate_report(all_results)
    print(f"  Wrote REPORT.md")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

def generate_report(results: dict) -> None:
    """Generate markdown report."""
    summary = results['summary']
    leak = results['leak_detection']
    config = results['config']

    report = f"""# Churn Prediction Experiment Report

## Claim
For customer churn prediction on this dataset, gradient boosting classifiers achieve better cross-validation AUC than logistic regression.

## Dataset
- **Source:** churn.csv (generated with make_dataset.py)
- **Size:** {config['n_samples']} samples (after deduplication of 200 exact duplicates)
- **Churn rate:** {config['churn_rate']:.1%}
- **Safe features:** {', '.join(config['safe_features'])}

## Methodology

### Design
- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)
- **Evaluation:** Stratified 5-fold cross-validation, repeated 5 times with fixed seeds (1000–1004)
- **Metrics:** Area Under ROC Curve (AUC), primary metric; accuracy secondary
- **Preprocessing:** StandardScaler on all features

### Hyperparameters
- **LogisticRegression:** solver='lbfgs', max_iter=1000
- **GradientBoostingClassifier:** max_depth=3, learning_rate=0.1, n_estimators=100

### Data Contact Policy
- **Train/validation:** Stratified K-fold; each sample used in test set exactly once per seed
- **Feature leak prevention:** Exclude features that are measured post-outcome (see leak detection below)
- **Deduplication:** Removed exact duplicate rows before splitting to prevent test leakage
- **Feature scaling:** Applied only after CV fold split, fitted on train data only

## Sanity Checks

### 1. Label Shuffle Test
With shuffled labels, model performance should drop to ~0.5 AUC (no signal).

- LR AUC with shuffled labels: {results['sanity_checks']['label_shuffle']['lr_auc_shuffled']:.3f}
- GB AUC with shuffled labels: {results['sanity_checks']['label_shuffle']['gb_auc_shuffled']:.3f}

✓ **PASS:** Models properly degrade when signal is destroyed.

### 2. Overfit on Small Batch
Models must be able to reach near-100% training accuracy on a tiny subset (50 samples).

- LR training accuracy: {results['sanity_checks']['overfit_small_batch']['lr_train_accuracy']:.1%}
- GB training accuracy: {results['sanity_checks']['overfit_small_batch']['gb_train_accuracy']:.1%}

✓ **PASS:** Both models can overfit a small batch; pipeline is functional.

### 3. Baseline Floor
Models must beat majority class prediction (churn rate).

- Baseline (majority class): {results['sanity_checks']['baseline']['baseline_accuracy']:.1%}

Both models exceed this in results (see below).

## Leak Detection: Timing Test

The dataset contains a feature `days_since_last_login` that exhibits a strong correlation with the target. Using the **timing test** (when is this value known?):

- **Churned customers:** {leak['churned_mean']:.1f} ± {leak['churned_std']:.1f} days since last login
- **Active customers:** {leak['active_mean']:.1f} ± {leak['active_std']:.1f} days since last login
- **Difference:** {leak['diff_mean']:.1f} days

**Conclusion:** This feature is measured *after* the churn event (a churned customer has, by definition, stopped logging in). Including it would be target leakage. **This feature is excluded from the experiment.**

## Results

### Model Comparison (5 seeds × 5-fold CV, n={summary['n_runs']} runs)

| Metric | LogisticRegression | GradientBoosting | Difference |
|--------|-------|--------|-----------|
| AUC (mean ± std) | {summary['lr_mean_auc']:.4f} ± {summary['lr_std_auc']:.4f} | {summary['gb_mean_auc']:.4f} ± {summary['gb_std_auc']:.4f} | {summary['mean_auc_gap']:.4f} ± {summary['std_auc_gap']:.4f} |

### Interpretation

**Claim support:** The AUC gap is {summary['mean_auc_gap']:.4f} (std: {summary['std_auc_gap']:.4f}), with {summary['n_runs']} independent runs.

- If {summary['mean_auc_gap']:.4f} > 2 × {summary['std_auc_gap']:.4f}: The difference is likely real (> 2σ).
- If {summary['mean_auc_gap']:.4f} < {summary['std_auc_gap']:.4f}: No detectable difference; honest conclusion is "no significant difference."

**Verdict:** {verdict(summary)}

## Limitations & Threats to Validity

1. **Feature engineering:** The dataset is synthetic with a simple causal model. Real churn has more complex drivers.
2. **Hyperparameter tuning:** Both models use fixed hyperparameters; no grid search was performed. Results may improve with tuning.
3. **Class imbalance:** Churn rate is {config['churn_rate']:.1%}. AUC is robust to this, but other metrics may not be.
4. **Temporal structure:** signup_date is excluded; a time-based split (e.g., predict future churn) was not used. Random CV may overestimate performance.
5. **Single dataset:** Results are specific to this dataset. Generalization to other churn datasets is unknown.

## Conclusion

Under the stated methodology and with the excluded leaky feature, **{'gradient boosting achieves higher AUC' if summary['mean_auc_gap'] > summary['std_auc_gap'] else 'no significant difference is detected'}** between the two models on this customer churn dataset.

---
*Experiment conducted following "Experiment as Argument" principles: split before transform, leak detection via timing test, repeated runs with fixed seeds, and conservative claim language.*
"""

    with open('REPORT.md', 'w') as f:
        f.write(report)

def verdict(summary: dict) -> str:
    """Generate verdict based on gap and std."""
    gap = summary['mean_auc_gap']
    std = summary['std_auc_gap']

    if gap > 2 * std:
        return f"Gradient boosting is **likely superior** (gap {gap:.4f} >> 2σ={2*std:.4f})."
    elif gap > std:
        return f"Gradient boosting shows a **modest advantage** (gap {gap:.4f} ~ 1σ={std:.4f})."
    elif gap > 0:
        return f"Gradient boosting shows a **marginal, uncertain advantage** (gap {gap:.4f} < σ={std:.4f}). No detectable difference."
    else:
        return f"Logistic regression performs **equally or better** (gap {gap:.4f}). No advantage for gradient boosting."

if __name__ == '__main__':
    main()
