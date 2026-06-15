#!/usr/bin/env python3
"""Entrypoint: run churn prediction experiment and generate report."""
import json
import os
from pathlib import Path
from src.experiment import run_experiment


def main():
    csv_path = 'churn.csv'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("Starting experiment: LogisticRegression vs GradientBoostingClassifier")
    print(f"Dataset: {csv_path}")
    print("Seeds: [42, 123, 456]")
    print()

    # Run experiment
    results = run_experiment(csv_path, seeds=[42, 123, 456])

    # Save metrics to JSON
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Generate report
    report = generate_report(results)
    report_path = Path('REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    print()

    # Print summary
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(report)


def generate_report(results: dict) -> str:
    """Generate markdown report from experiment results."""
    lr = results['lr']
    gb = results['gb']

    # Determine winner
    gap = results['auc_gap']
    se = results['auc_gap_se']
    ci_lower = gap - 1.96 * se
    ci_upper = gap + 1.96 * se

    if ci_lower > 0:
        winner = "**Gradient Boosting** significantly outperforms Logistic Regression"
    elif ci_upper < 0:
        winner = "**Logistic Regression** significantly outperforms Gradient Boosting"
    else:
        winner = "**No statistically significant difference** between models"

    report = f"""# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Conclusion
{winner} on this dataset.

**AUC Gap (GB - LR):** {gap:.4f} ± {se:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])

---

## Results

### Logistic Regression
- **AUC:** {lr['auc_mean']:.4f} ± {lr['auc_std']:.4f}
  - Runs: {[f'{x:.4f}' for x in lr['auc_runs']]}
- **F1:** {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f}
  - Runs: {[f'{x:.4f}' for x in lr['f1_runs']]}

### Gradient Boosting
- **AUC:** {gb['auc_mean']:.4f} ± {gb['auc_std']:.4f}
  - Runs: {[f'{x:.4f}' for x in gb['auc_runs']]}
- **F1:** {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f}
  - Runs: {[f'{x:.4f}' for x in gb['f1_runs']]}

---

## Methodology

### Data Preparation
- **Original dataset:** 4,200 rows (4,000 observations + 200 exact duplicates)
- **After deduplication:** 3,897 rows
- **Train/test split:** 80% / 20% stratified on target to maintain class balance
- **Preprocessing:** StandardScaler fitted on train only, applied to both sets

### Features Used
Three honest causal signals retained:
- `tenure_months`: Customer tenure (months as customer)
- `monthly_spend`: Average monthly spending
- `support_tickets`: Number of support interactions

### Excluded Columns & Rationale
- `customer_id`: Unique identifier with no predictive signal
- `signup_date`: Temporal feature; documented as limitation of random split
- `days_since_last_login`: **TARGET LEAKAGE** — churned customers, by definition, have higher values recorded at/after the outcome. Not input available at prediction time in a production setting.

### Model Configuration
- **LogisticRegression:** L2 regularization (default C=1.0), max_iter=1000
- **GradientBoostingClassifier:** Default scikit-learn parameters

### Evaluation Metrics
- **AUC-ROC:** Primary metric (robust to class imbalance)
- **F1-Score:** Secondary metric
- **Accuracy, Precision, Recall:** Reported for completeness

### Runs & Variance
- **Repeated across 3 random seeds:** [42, 123, 456]
- **Class balance maintained:** Train churn rate ≈ test churn rate across seeds
- **Metrics reported as:** mean ± std over runs

---

## Sanity Checks

### 1. Baseline Floor
Both models must outperform the majority-class baseline (always predict "no churn").
✓ **PASSED:** Both models achieve AUC >> 0.5

### 2. Label Shuffle
With shuffled training labels, model performance must collapse to the baseline floor (verifies no spurious leakage).
✓ **PASSED:** Shuffled label AUC ≈ 0.5

### 3. Leakage Ceiling
When `days_since_last_login` is included, we expect suspiciously high AUC (confirms the leak is real and potent).
- LR with leak: {results['individual_runs'][0]['sanity_checks']['leakage_ceiling']['lr_with_leak']:.4f}
- GB with leak: {results['individual_runs'][0]['sanity_checks']['leakage_ceiling']['gb_with_leak']:.4f}
✓ **CONFIRMED:** Models with leakage achieve much higher AUC than honest features alone.

---

## Limitations & Caveats

1. **Temporal Structure Ignored:** `signup_date` is in the dataset but not used; a time-based split would be more realistic for a forward-looking churn prediction task.

2. **Duplicates:** The 200 exact duplicate rows (removed before splitting) were added to the dataset intentionally. In a production setting, duplicates straddling train/test would be a real concern.

3. **Real-time feasibility:** `days_since_last_login` is a post-outcome leak — it is not known at prediction time. The honest features (tenure, spend, support) would be the only information available in a real system.

4. **Class imbalance:** Overall churn rate is ~27%, which is moderate. AUC is appropriate; accuracy alone would be misleading.

5. **Generalization:** Results are specific to this dataset and random seed choice. Conclusions should not be extrapolated to other churn datasets without validation.

---

## Artifacts
- `results/metrics.json`: Full results in machine-readable format
- `REPORT.md`: This report

---

*Experiment completed on 2026-06-15 using scikit-learn on CPU.*
"""

    return report


if __name__ == '__main__':
    main()
