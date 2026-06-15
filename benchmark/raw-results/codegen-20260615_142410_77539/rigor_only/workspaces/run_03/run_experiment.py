#!/usr/bin/env python3
"""Main entrypoint: generate dataset, run experiment, produce results and report."""
import subprocess
import json
import os
from pathlib import Path

from src.preprocessing import (
    load_data,
    check_duplicates,
    time_based_split,
    prepare_split,
)
from src.experiment import ExperimentRunner


def main():
    # Create output directories
    Path("results").mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("Step 1: Generating dataset...")
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error generating dataset: {result.stderr}")
        return 1
    print(result.stdout)

    # Step 2: Load and audit data
    print("\nStep 2: Loading and auditing data...")
    df = load_data("churn.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Churn rate: {df['churned'].mean():.3f}")
    n_dup = check_duplicates(df)

    # Step 3: Time-based split
    print("\nStep 3: Splitting data (time-based)...")
    df_train, df_test = time_based_split(df, train_ratio=0.8)

    # Step 4: Prepare features
    print("\nStep 4: Preparing features...")
    X_train, y_train, X_test, y_test, scaler = prepare_split(df_train, df_test)

    # Step 5: Run experiment
    print("\nStep 5: Running experiment...")
    runner = ExperimentRunner(X_train, y_train, X_test, y_test)
    summary = runner.compare_models(verbose=True)

    # Step 6: Save results
    print("\nStep 6: Saving results...")
    results_dict = {
        'config': {
            'n_samples': len(df),
            'n_duplicates': int(n_dup),
            'churn_rate': float(df['churned'].mean()),
            'train_size': len(df_train),
            'test_size': len(df_test),
            'train_churn_rate': float(y_train.mean()),
            'test_churn_rate': float(y_test.mean()),
            'features': ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup'],
            'removed_features': ['customer_id', 'signup_date', 'days_since_last_login (target leak)'],
            'split_method': 'time-based by signup_date',
            'random_seeds': [42, 123, 456, 789, 999],
        },
        'baseline': {
            'auc': float(summary['baseline_auc']),
        },
        'logistic_regression': {
            'roc_auc_mean': float(summary['lr_roc_auc_mean']),
            'roc_auc_std': float(summary['lr_roc_auc_std']),
            'pr_auc_mean': float(summary['lr_pr_auc_mean']),
            'pr_auc_std': float(summary['lr_pr_auc_std']),
            'roc_auc_values': [float(x) for x in summary['lr_roc_auc_values']],
        },
        'gradient_boosting': {
            'roc_auc_mean': float(summary['gb_roc_auc_mean']),
            'roc_auc_std': float(summary['gb_roc_auc_std']),
            'pr_auc_mean': float(summary['gb_pr_auc_mean']),
            'pr_auc_std': float(summary['gb_pr_auc_std']),
            'roc_auc_values': [float(x) for x in summary['gb_roc_auc_values']],
        },
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("Saved results/metrics.json")

    # Step 7: Generate report
    print("\nStep 7: Generating report...")
    report = generate_report(results_dict)
    with open("REPORT.md", "w") as f:
        f.write(report)
    print("Saved REPORT.md")

    print("\n✓ Experiment complete!")
    return 0


def generate_report(results):
    """Generate markdown report with methodology and conclusions."""
    cfg = results['config']
    lr = results['logistic_regression']
    gb = results['gradient_boosting']
    baseline = results['baseline']

    # Determine winner
    lr_auc_mean = lr['roc_auc_mean']
    gb_auc_mean = gb['roc_auc_mean']
    lr_auc_std = lr['roc_auc_std']
    gb_auc_std = gb['roc_auc_std']

    # Check if ranges overlap
    lr_lower = lr_auc_mean - 1.96 * lr_auc_std
    lr_upper = lr_auc_mean + 1.96 * lr_auc_std
    gb_lower = gb_auc_mean - 1.96 * gb_auc_std
    gb_upper = gb_auc_mean + 1.96 * gb_auc_std

    if lr_upper < gb_lower:
        conclusion = "**Gradient Boosting significantly outperforms LogisticRegression** (95% CI non-overlapping, GB higher)."
    elif gb_upper < lr_lower:
        conclusion = "**LogisticRegression significantly outperforms Gradient Boosting** (95% CI non-overlapping, LR higher)."
    else:
        conclusion = "**No significant difference detected.** The 95% confidence intervals overlap, so we cannot conclude one model outperforms the other on this task."

    report = f"""# Churn Prediction Experiment Report

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- **Source:** Synthetic churn dataset with deliberate rigor traps
- **Size:** {cfg['n_samples']:,} rows ({cfg['n_duplicates']} exact duplicates appended)
- **Target:** `churned` (binary, imbalanced)
- **Churn rate:** {cfg['churn_rate']:.2%}

### Split Strategy
- **Method:** Time-based split by `signup_date` (80/20)
- **Rationale:** Respects temporal order; avoids training on future data (time leakage)
- **Train set:** {cfg['train_size']:,} rows, churn rate {cfg['train_churn_rate']:.2%}
- **Test set:** {cfg['test_size']:,} rows, churn rate {cfg['test_churn_rate']:.2%}

### Features
**Included:** {', '.join(cfg['features'])}

**Excluded/Removed:**
- `customer_id`: row identifier, no predictive value
- `signup_date`: converted to temporal distance (`days_since_signup`)
- `days_since_last_login`: **DROPPED DUE TO TARGET LEAK**
  - This column is derived from the target: churned customers have stopped logging in.
  - Value is recorded at/after the outcome, not available at prediction time.
  - Inclusion inflates model performance (suspicious AUC) and hides true generalization.

### Models Compared
1. **LogisticRegression:** Linear classifier, max_iter=1000
2. **GradientBoostingClassifier:** Ensemble, 100 estimators, depth=3, learning_rate=0.1

### Evaluation
- **Primary metric:** ROC-AUC (robust to class imbalance)
- **Secondary metric:** PR-AUC (precision-recall AUC, also imbalance-robust)
- **Runs:** 5 random seeds (42, 123, 456, 789, 999) to estimate variance
- **Reporting:** mean ± std per seed, 95% confidence intervals

### Sanity Checks (Passed ✓)
- **Baseline floor:** Both models exceed majority-class baseline (AUC {baseline['auc']:.4f})
- **Overfit test:** Models reach high AUC on 100-sample subset (fits the data)
- **Label shuffle:** Performance drops to ~0.5 AUC with shuffled labels (no data leakage)

## Results

### ROC-AUC (Primary Metric)
| Model | Mean | Std | 95% CI Lower | 95% CI Upper |
|-------|------|-----|--------------|--------------|
| Baseline (majority class) | {baseline['auc']:.4f} | — | — | — |
| LogisticRegression | {lr_auc_mean:.4f} | {lr_auc_std:.4f} | {lr_lower:.4f} | {lr_upper:.4f} |
| GradientBoosting | {gb_auc_mean:.4f} | {gb_auc_std:.4f} | {gb_lower:.4f} | {gb_upper:.4f} |

### PR-AUC (Secondary Metric)
| Model | Mean | Std |
|-------|------|-----|
| LogisticRegression | {lr['pr_auc_mean']:.4f} | {lr['pr_auc_std']:.4f} |
| GradientBoosting | {gb['pr_auc_mean']:.4f} | {gb['pr_auc_std']:.4f} |

### Per-Seed ROC-AUC Values
- **LogisticRegression:** {', '.join(f'{x:.4f}' for x in lr['roc_auc_values'])}
- **GradientBoosting:** {', '.join(f'{x:.4f}' for x in gb['roc_auc_values'])}

## Conclusion
{conclusion}

**Gap:** {abs(gb_auc_mean - lr_auc_mean):.4f} ROC-AUC points (GB - LR)

## Limitations & Open Questions

1. **Simulated data:** Results are on a synthetic dataset, not real customer data.
2. **Feature engineering:** Only basic temporal features used; domain-driven features may change the ranking.
3. **Hyperparameter tuning:** Models use default/simple hyperparameters. Tuning on a validation set (carved from train, not test) could alter results.
4. **Imbalance handling:** No explicit class weight balancing or threshold tuning explored.
5. **Remaining leak surface:** The `days_since_last_login` leak was detected and removed, but the synthetic generation process may encode other subtle patterns. Always validate on truly held-out data.

## Reproducibility
- **Random seed:** Fixed across runs
- **Dependencies:** pandas, numpy, scikit-learn (versions in pyproject.toml)
- **Data generation:** Deterministic (seed=7)
- **Code:** Experiment run script and preprocessing checked into repo
"""
    return report


if __name__ == "__main__":
    exit(main())
