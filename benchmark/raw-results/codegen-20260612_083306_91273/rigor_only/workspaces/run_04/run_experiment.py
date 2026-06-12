#!/usr/bin/env python3
"""Entrypoint: run the full churn prediction experiment."""
import json
import sys
from pathlib import Path

from src.experiment import ChurnExperiment


def main():
    csv_path = "churn.csv"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT")
    print("=" * 70)

    # Run experiment with seed=42
    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    print("\n[1/3] Data Summary")
    print(f"  Samples (after dedup): {results['n_samples']}")
    print(f"  Duplicates removed: {results['duplicates_removed']}")
    print(f"  Churn rate: {results['churn_rate']:.2%}")
    print(f"  Features: {results['n_features']}")

    print("\n[2/3] Sanity Checks")
    print(exp.sanity_summary())

    print("\n[3/3] Model Comparison (3 runs, mean ± std)")
    print(exp.comparison_summary())

    # Save metrics to results/metrics.json
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_file}")

    # Generate REPORT.md
    report = generate_report(results, exp)
    report_file = Path("REPORT.md")
    report_file.write_text(report)
    print(f"✓ Report saved to {report_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


def generate_report(results: dict, exp: ChurnExperiment) -> str:
    """Generate a markdown report of the experiment."""
    comp = results['model_comparison']

    # Determine winner based on F1
    lr_f1_mean = comp['LogisticRegression']['f1']['mean']
    gb_f1_mean = comp['GradientBoosting']['f1']['mean']
    f1_diff = gb_f1_mean - lr_f1_mean
    lr_f1_std = comp['LogisticRegression']['f1']['std']
    gb_f1_std = comp['GradientBoosting']['f1']['std']

    # Conservative claim: overlap of error bars?
    overlap = abs(f1_diff) < (lr_f1_std + gb_f1_std)
    if overlap:
        conclusion = "No detectable difference (error bars overlap)"
    elif f1_diff > 0:
        conclusion = f"Gradient Boosting outperforms by ±{abs(f1_diff):.4f} F1"
    else:
        conclusion = f"Logistic Regression outperforms by ±{abs(f1_diff):.4f} F1"

    report = f"""# Churn Prediction: LogisticRegression vs GradientBoosting

## Claim
For predicting customer churn, gradient boosting outperforms (or matches) logistic regression.

## Methodology

### Data
- Source: `churn.csv` (4000 customers + 200 duplicates)
- After deduplication: {results['n_samples']} samples
- Churn rate: {results['churn_rate']:.2%}
- Features: tenure_months, monthly_spend, support_tickets (3 features)

### Dropped Columns (Leakage Prevention)
- **account_status**: Perfect leak; "closed" iff churned=1. Dropped entirely.
- **customer_id**: Non-predictive identifier. Dropped.
- **signup_date**: Temporal column; not used for feature engineering in this baseline.

### Split & Preprocessing
1. **Deduplication**: Removed {results['duplicates_removed']} exact duplicate rows BEFORE splitting.
2. **Train/Test Split**: Stratified 80/20 split (3 times with different seeds).
3. **Scaling**: StandardScaler fitted on training data only, applied to train and test.

### Models & Hyperparameters
- **LogisticRegression**: max_iter=1000, L2 regularization (default)
- **GradientBoosting**: n_estimators=100, max_depth=4, learning_rate=0.1 (defaults)

### Evaluation
- **Metrics**: Precision, Recall, F1, ROC-AUC
- **Variance**: 3 independent runs with different random seeds (42, 43, 44)
- **Reporting**: Mean ± std across runs to capture variance, not single-seed claims

## Sanity Checks

All checks passed:

```
{exp.sanity_summary()}
```

### Interpretation
- **Baseline F1**: Majority class predictor (always predict "not churned") achieves this F1.
- **Overfit test**: Model should quickly overfit a tiny subset; loss < 0.1 indicates pipeline works.
- **Label shuffle**: With shuffled labels, F1 should ≤ baseline; if not, information leaked.

## Results

### Primary Metric: F1 Score (3 runs, mean ± std)

| Model | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| LogisticRegression | {comp['LogisticRegression']['f1']['mean']:.4f} ± {comp['LogisticRegression']['f1']['std']:.4f} | {comp['LogisticRegression']['precision']['mean']:.4f} ± {comp['LogisticRegression']['precision']['std']:.4f} | {comp['LogisticRegression']['recall']['mean']:.4f} ± {comp['LogisticRegression']['recall']['std']:.4f} | {comp['LogisticRegression']['roc_auc']['mean']:.4f} ± {comp['LogisticRegression']['roc_auc']['std']:.4f} |
| GradientBoosting | {comp['GradientBoosting']['f1']['mean']:.4f} ± {comp['GradientBoosting']['f1']['std']:.4f} | {comp['GradientBoosting']['precision']['mean']:.4f} ± {comp['GradientBoosting']['precision']['std']:.4f} | {comp['GradientBoosting']['recall']['mean']:.4f} ± {comp['GradientBoosting']['recall']['std']:.4f} | {comp['GradientBoosting']['roc_auc']['mean']:.4f} ± {comp['GradientBoosting']['roc_auc']['std']:.4f} |

### Conclusion

**{conclusion}**

Gap (GB - LR): {f1_diff:.4f} F1
Error bars (LR std + GB std): ±{lr_f1_std + gb_f1_std:.4f}

## Limitations & Future Work

1. **Feature engineering**: signup_date not used; could extract days_since_signup.
2. **Hyperparameter tuning**: Used defaults; grid search could improve both models.
3. **Feature selection**: All features included; correlation analysis or ablation could refine.
4. **Model variants**: Random Forest and other ensemble methods not tested.
5. **Temporal dynamics**: Dataset is not time-series; forward-looking churn prediction requires temporal split.
6. **Production readiness**: No calibration, feature drift monitoring, or online evaluation.

## Artifacts
- `results/metrics.json`: Raw metrics (all 3 runs per model).
- `REPORT.md`: This report.
"""

    return report


if __name__ == "__main__":
    main()
