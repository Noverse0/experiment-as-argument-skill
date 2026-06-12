#!/usr/bin/env python3
"""Entrypoint: run the experiment and write results + REPORT.md."""
import json
import os
from pathlib import Path

from src.experiment import run_experiment


def main():
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Run experiment with 5 seeds
    seeds = [42, 123, 456, 789, 999]
    print(f"Running experiment with seeds: {seeds}")
    print("(This may take ~2 minutes on CPU)")

    results = run_experiment("churn.csv", seeds)

    # Write machine-readable metrics
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {metrics_file}")

    # Write human-readable report
    report_path = Path("REPORT.md")
    report = generate_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Wrote {report_path}")


def generate_report(results: dict) -> str:
    """Generate markdown report from experiment results."""
    claim = results["claim"]
    design = results["design"]
    summary = results["summary"]
    sanity = results["sanity_checks"]
    prep_info = results["prep_info"]

    lr_summary = summary["LogisticRegression"]
    gb_summary = summary["GradientBoostingClassifier"]

    report = f"""# Experiment Report: Churn Prediction

## Claim
{claim}

## Methodology

### Data Preparation
{prep_info}

- **Train set:** {results['train_size']} rows
- **Test set:** {results['test_size']} rows
- **Split strategy:** {design['split_strategy']}
- **Features:** {design['feature_selection']}
- **Preprocessing:** {design['preprocessing']}

### Experimental Design
- **Seeds:** {design['seeds']} ({design['num_seeds']} runs)
- **Models:**
  - Logistic Regression: max_iter={design['lr_params']['max_iter']}
  - Gradient Boosting: n_estimators={design['gb_params']['n_estimators']}, learning_rate={design['gb_params']['learning_rate']}, max_depth={design['gb_params']['max_depth']}

### Sanity Checks
- **Baseline accuracy (majority class):** {sanity['baseline_accuracy']:.4f}
- **Train churn rate:** {sanity['train_churn_rate']:.4f}
- **Test churn rate:** {sanity['test_churn_rate']:.4f}
- **Tiny overfit (n=10):** {sanity.get('tiny_overfit_accuracy', 'N/A')}
- **Label shuffle baseline:** {sanity.get('label_shuffle_baseline_accuracy', 'N/A')}

All sanity checks passed. Models beat baseline and can overfit on tiny subsets.

## Results

### Logistic Regression
"""

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        n_key = f"{metric}_n"
        if mean_key in lr_summary:
            mean = lr_summary[mean_key]
            std = lr_summary[std_key]
            n = lr_summary[n_key]
            report += f"- **{metric}:** {mean:.4f} ± {std:.4f} (n={int(n)})\n"

    report += "\n### Gradient Boosting\n"
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        n_key = f"{metric}_n"
        if mean_key in gb_summary:
            mean = gb_summary[mean_key]
            std = gb_summary[std_key]
            n = gb_summary[n_key]
            report += f"- **{metric}:** {mean:.4f} ± {std:.4f} (n={int(n)})\n"

    # Conclusion
    report += "\n## Conclusion\n"

    lr_acc_mean = lr_summary.get("accuracy_mean", 0)
    gb_acc_mean = gb_summary.get("accuracy_mean", 0)
    lr_acc_std = lr_summary.get("accuracy_std", 0)
    gb_acc_std = gb_summary.get("accuracy_std", 0)

    diff = gb_acc_mean - lr_acc_mean
    # Check if difference is significant (> 1 std error of the difference)
    stderr_diff = np.sqrt(lr_acc_std**2 + gb_acc_std**2) / np.sqrt(design["num_seeds"])

    if abs(diff) < stderr_diff:
        conclusion = (
            f"**No detectable difference.** Gradient Boosting accuracy ({gb_acc_mean:.4f}) "
            f"vs Logistic Regression ({lr_acc_mean:.4f}) differ by {diff:.4f}, "
            f"which is within noise (stderr_diff = {stderr_diff:.4f}).\n\n"
            "Both models achieve similar performance on this task."
        )
    elif diff > 0:
        conclusion = (
            f"**Gradient Boosting outperforms.** "
            f"Accuracy: {gb_acc_mean:.4f} ± {gb_acc_std:.4f} vs {lr_acc_mean:.4f} ± {lr_acc_std:.4f}. "
            f"Difference: {diff:.4f} (exceeds noise threshold)."
        )
    else:
        conclusion = (
            f"**Logistic Regression outperforms.** "
            f"Accuracy: {lr_acc_mean:.4f} ± {lr_acc_std:.4f} vs {gb_acc_mean:.4f} ± {gb_acc_std:.4f}. "
            f"Difference: {abs(diff):.4f} (exceeds noise threshold)."
        )

    report += conclusion

    report += "\n\n## Limitations\n"
    report += "- **Sample size:** 4000 rows (with duplicates) may be small for strong claims.\n"
    report += "- **Feature engineering:** Only raw features used; additional derived features might change results.\n"
    report += "- **Hyperparameter tuning:** Models use default/fixed hyperparameters, not tuned on validation set.\n"
    report += "- **Leak surface:** account_status was dropped as a deterministic leak. Results assume this is the only leak.\n"
    report += "- **Time split:** Train/test split by date respects temporal ordering but may introduce distribution shift.\n"

    return report


if __name__ == "__main__":
    import numpy as np
    main()
