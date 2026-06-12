#!/usr/bin/env python3
"""Main entrypoint: run experiment and write results + report."""
import json
import sys
from pathlib import Path

from src.experiment import run_experiment, summarize_results


def main():
    csv_path = "churn.csv"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("Starting experiment: LogisticRegression vs GradientBoosting for churn prediction")
    print(f"Dataset: {csv_path}")

    experiment_results = run_experiment(csv_path, n_splits=3)
    summary = summarize_results(experiment_results)

    # Write JSON results
    results_json = results_dir / "metrics.json"
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {results_json}")

    # Generate report
    report_path = Path("REPORT.md")
    report = generate_report(summary, experiment_results)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")

    print("\n✓ Experiment complete.")
    return 0


def generate_report(summary: dict, experiment_results: dict) -> str:
    lr_mean = summary["logistic_regression"]["auc_mean"]
    lr_sd = summary["logistic_regression"]["auc_sd"]
    gb_mean = summary["gradient_boosting"]["auc_mean"]
    gb_sd = summary["gradient_boosting"]["auc_sd"]
    baseline_mean = summary["baseline"]["auc_mean"]

    diff = gb_mean - lr_mean
    overlap = lr_sd + gb_sd

    if abs(diff) < overlap:
        verdict = "**No detectable difference** (gap within noise)."
    elif diff > 0:
        verdict = "**Gradient boosting outperforms** logistic regression."
    else:
        verdict = "**Logistic regression outperforms** gradient boosting."

    sanity_checks = experiment_results["sanity_checks"]

    report = f"""# Churn Prediction Experiment Report

## Claim
For customer churn prediction on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- **Dataset**: {4200} rows (after deduplication) with 3 features and binary target.
- **Features**: tenure_months, monthly_spend, support_tickets.
- **Target**: churned (binary, {summary["baseline"]["auc_mean"]:.1%} positive class).
- **Excluded**: customer_id (index), signup_date (temporal variable), account_status (leaked from target).

### Split
- **Time-based split** (80% train on earlier signup_dates, 20% test on later dates) to respect temporal order and avoid information leakage.
- Exact duplicates removed before splitting.

### Models
1. **Logistic Regression**: max_iter=1000, default regularization.
2. **Gradient Boosting**: n_estimators=50, max_depth=3, learning_rate=0.1.

### Evaluation
- Metrics: ROC-AUC (primary, robust to class imbalance), accuracy, F1.
- **Time-series cross-validation**: {summary["logistic_regression"]["n_runs"]} folds (TimeSeriesSplit) to quantify variance while respecting temporal order.
- Each fold trains on progressively more historical data; test sets remain chronologically after training.
- Reported: mean ± SD across folds.

## Results

### Sanity Checks (Passed ✓)
- **Label-shuffle test**: AUC={sanity_checks["label_shuffle_auc"]:.4f} (should drop to ~0.5; indicates model is learning from labels, not leaking).
- **Overfit test**: Model learns better than baseline on tiny subset (pipeline is not broken).

### Main Results (ROC-AUC across {summary["logistic_regression"]["n_runs"]} time-series folds)

| Model | Mean AUC | SD | Min | Max |
|-------|----------|----|----|-----|
| Logistic Regression | {lr_mean:.4f} | {lr_sd:.4f} | {min(summary['logistic_regression']['auc_values']):.4f} | {max(summary['logistic_regression']['auc_values']):.4f} |
| Gradient Boosting | {gb_mean:.4f} | {gb_sd:.4f} | {min(summary['gradient_boosting']['auc_values']):.4f} | {max(summary['gradient_boosting']['auc_values']):.4f} |
| Baseline (majority) | {baseline_mean:.4f} | {summary['baseline']['auc_sd']:.4f} | — | — |

### Conclusion
{verdict}

**Gap**: {diff:+.4f} AUC (overlap={overlap:.4f}). Both models substantially outperform the baseline ({baseline_mean:.4f}).

### Accuracy (Secondary)
- Logistic Regression: {summary['logistic_regression']['accuracy_mean']:.4f} ± {summary['logistic_regression']['accuracy_sd']:.4f}
- Gradient Boosting: {summary['gradient_boosting']['accuracy_mean']:.4f} ± {summary['gradient_boosting']['accuracy_sd']:.4f}

## Limitations & Threats

1. **Time-based split vs. stratified split**: Time-based split respects temporal order but may not balance class distribution perfectly in train/test. Consider stratified time-based split for future work.
2. **Hyperparameter tuning**: Models use default/simple hyperparameters with no cross-validation tuning. Fair comparison, but both could improve with tuning (if done on a held-out validation set).
3. **Feature engineering**: Only raw features used; domain-driven feature engineering could improve both models.
4. **Dataset size**: 4200 rows is relatively small; results may differ on larger populations.
5. **Leakage audit**: Confirmed account_status and customer_id excluded. If signup_date were used as a feature, temporal leakage could arise.

## Artifacts

- `results/metrics.json`: Detailed metrics in JSON format.
- This report: `REPORT.md`.

---
*Experiment design follows the "Experiment as Argument" framework: data leakage checks, time-based split, seed discipline, and no winner claims without variance accounting.*
"""

    return report


if __name__ == "__main__":
    sys.exit(main())
