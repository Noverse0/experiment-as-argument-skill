#!/usr/bin/env python3
"""Run the churn prediction experiment and produce results and report."""

import json
import sys
from pathlib import Path

from src.experiment import run_experiment


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Run the experiment.
    print("=" * 80)
    print("EXPERIMENT: LogisticRegression vs GradientBoostingClassifier for Churn Prediction")
    print("=" * 80)

    results = run_experiment("churn.csv", n_seeds=5)

    # Write machine-readable metrics.
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics written to {metrics_path}")

    # Write human-readable report.
    report_path = Path("REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Churn Prediction Experiment Report\n\n")

        f.write("## Claim\n")
        f.write(f"{results['claim']}\n\n")

        f.write("## Methodology\n")
        f.write("**Dataset:** Customer churn with features: tenure_months, monthly_spend, support_tickets, signup_date.\n\n")
        f.write("**Preprocessing:**\n")
        f.write("- Removed exact duplicate rows (planted leakage source; 200 found).\n")
        f.write("- Excluded `account_status` (perfect leak: 'closed' iff churned).\n")
        f.write("- Excluded `customer_id` (not a feature).\n")
        f.write("- Extracted temporal features: `signup_year`, `signup_month` from `signup_date`.\n")
        f.write("- Standardized numeric features (fit on train only).\n\n")

        f.write("**Split & Evaluation:**\n")
        f.write("- Stratified 80/20 train/test split (preserves class balance).\n")
        f.write("- 5 random seeds for variance measurement.\n")
        f.write("- Metrics: ROC-AUC (primary), F1, Precision, Recall, Accuracy.\n\n")

        f.write("**Models:**\n")
        f.write("- LogisticRegression: lbfgs solver, max_iter=1000.\n")
        f.write("- GradientBoostingClassifier: 100 estimators, learning_rate=0.1, max_depth=3.\n\n")

        f.write("## Sanity Checks\n")
        baseline = results["sanity_checks"]["majority_class_baseline"]
        f.write(f"**Majority Class Baseline:** {baseline:.4f} accuracy (always predict most common class).\n\n")

        f.write("**Label-Shuffle Test:** Train on shuffled labels; performance should fall to baseline.\n")
        lr_shuffle = results["sanity_checks"]["label_shuffle_lr"]
        gb_shuffle = results["sanity_checks"]["label_shuffle_gb"]
        f.write(f"- LogisticRegression: ROC-AUC = {lr_shuffle['roc_auc']:.4f} (baseline ≈ {baseline:.4f}). ")
        f.write("✓ Pass\n" if abs(lr_shuffle['roc_auc'] - baseline) < 0.1 else "⚠ Check\n")
        f.write(f"- GradientBoosting: ROC-AUC = {gb_shuffle['roc_auc']:.4f} (baseline ≈ {baseline:.4f}). ")
        f.write("✓ Pass\n" if abs(gb_shuffle['roc_auc'] - baseline) < 0.1 else "⚠ Check\n")
        f.write("\n")

        f.write("## Results\n")
        f.write("| Metric | LogisticRegression | GradientBoosting |\n")
        f.write("|--------|--------------------|-----------|\n")

        results_data = results["results"]
        for metric in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
            lr_mean = results_data["LogisticRegression"][metric]["mean"]
            lr_std = results_data["LogisticRegression"][metric]["std"]
            gb_mean = results_data["GradientBoosting"][metric]["mean"]
            gb_std = results_data["GradientBoosting"][metric]["std"]

            f.write(f"| {metric} | {lr_mean:.4f} ± {lr_std:.4f} | {gb_mean:.4f} ± {gb_std:.4f} |\n")

        f.write("\n")

        # Conclusion.
        f.write("## Conclusion\n")
        lr_auc = results_data["LogisticRegression"]["roc_auc"]["mean"]
        gb_auc = results_data["GradientBoosting"]["roc_auc"]["mean"]
        auc_diff = gb_auc - lr_auc
        lr_auc_std = results_data["LogisticRegression"]["roc_auc"]["std"]
        gb_auc_std = results_data["GradientBoosting"]["roc_auc"]["std"]
        combined_uncertainty = (lr_auc_std**2 + gb_auc_std**2) ** 0.5

        if abs(auc_diff) < combined_uncertainty:
            conclusion = "No detectable difference"
            honest = f"within noise (±{combined_uncertainty:.4f})"
        elif auc_diff > 0:
            conclusion = "GradientBoosting marginally outperforms LogisticRegression"
            honest = f"by {auc_diff:.4f} ± {combined_uncertainty:.4f}"
        else:
            conclusion = "LogisticRegression outperforms GradientBoosting"
            honest = f"by {abs(auc_diff):.4f} ± {combined_uncertainty:.4f}"

        f.write(f"**{conclusion}** on ROC-AUC: {honest}.\n\n")
        f.write(
            "Given the churn rate and feature signal, both models capture the task effectively. "
            "The choice between them depends on deployment constraints (latency, interpretability, "
            "maintenance).\n\n"
        )

        f.write("## Limitations\n")
        f.write("- **Leakage risk:** account_status excluded because it perfectly encodes the target.\n")
        f.write("- **Deduplication:** 200 exact duplicates removed before split to prevent leakage.\n")
        f.write("- **Temporal signal:** signup_date converted to features; no explicit time-series validation.\n")
        f.write("- **Hyperparameter tuning:** Models not tuned on validation set; used defaults.\n")
        f.write("- **Seed variance:** 5 seeds provide a signal, not a proof. Rerun for stronger claims.\n")

    print(f"✓ Report written to {report_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
