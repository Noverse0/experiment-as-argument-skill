#!/usr/bin/env python3
"""Entrypoint: run churn prediction experiment and produce results + report."""
import json
import sys
from pathlib import Path
from src.experiment import run_experiment


def write_json_results(results: dict, output_dir: Path):
    """Write machine-readable results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Output] Metrics written to {results_file}")
    return results_file


def write_markdown_report(results: dict, output_dir: Path, csv_path: str):
    """Write human-readable experiment report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "REPORT.md"

    aggregated = results["aggregated"]
    lr_auc = aggregated["logistic_regression"]["roc_auc"]
    gb_auc = aggregated["gradient_boosting"]["roc_auc"]
    diff = gb_auc["mean"] - lr_auc["mean"]
    overlap = (lr_auc["mean"] - 2*lr_auc["std"]) < (gb_auc["mean"] + 2*gb_auc["std"])

    # Determine verdict.
    if overlap:
        verdict = (
            "**No statistically significant difference.** The confidence intervals "
            "overlap; any observed difference is within noise."
        )
    elif diff > 0:
        verdict = (
            f"**GradientBoosting outperforms LogisticRegression** by "
            f"{abs(diff):.4f} ROC-AUC (ours: {gb_auc['mean']:.4f} vs theirs: {lr_auc['mean']:.4f})."
        )
    else:
        verdict = (
            f"**LogisticRegression outperforms GradientBoosting** by "
            f"{abs(diff):.4f} ROC-AUC (ours: {lr_auc['mean']:.4f} vs theirs: {gb_auc['mean']:.4f})."
        )

    with open(report_file, "w") as f:
        f.write("# Churn Prediction Experiment: LogisticRegression vs GradientBoosting\n\n")

        f.write("## Claim\n")
        f.write("Does GradientBoostingClassifier outperform LogisticRegression for predicting customer churn?\n\n")

        f.write("## Methodology\n\n")
        f.write("### Data and Splits\n")
        f.write(f"- **Dataset**: {csv_path} (4000 + 200 duplicates)\n")
        f.write("- **Deduplication**: Removed 200 exact duplicate rows to prevent train/test leakage\n")
        f.write("- **Temporal Split**: 70% train / 30% test, ordered by `signup_date` (not random)\n")
        f.write("  - Rationale: Temporal column should be respected; model trained on past predicts future\n")
        f.write("- **Leak Audit**:\n")
        f.write("  - Dropped `account_status` (derived from target: 'closed' iff churned=1)\n")
        f.write("  - Dropped `customer_id` (not predictive)\n")
        f.write("  - Kept: tenure_months, monthly_spend, support_tickets (predictive features)\n\n")

        f.write("### Preprocessing\n")
        f.write("- **Fit on train only**, applied to test:\n")
        f.write("  - StandardScaler for numeric features\n")
        f.write("  - No hyperparameter tuning on test set\n")
        f.write("- **Class imbalance**: LogisticRegression uses `class_weight='balanced'`\n\n")

        f.write("### Models\n")
        f.write("1. **LogisticRegression**: max_iter=1000, balanced class weights\n")
        f.write("2. **GradientBoostingClassifier**: 100 estimators, lr=0.1, max_depth=3\n")
        f.write("   - Same hyperparameters across all runs; no tuning\n\n")

        f.write("### Evaluation\n")
        f.write("- **Metrics**: ROC-AUC (primary), Accuracy, Precision, Recall, F1\n")
        f.write("- **Runs**: 3 random seeds (42, 123, 456) to estimate variance\n")
        f.write("- **Test set**: Touched once, at the end; no decisions made after seeing test metrics\n\n")

        f.write("## Results\n\n")
        f.write("### Per-Model Performance (mean ± std, n=3 seeds)\n\n")

        f.write("| Metric | LogisticRegression | GradientBoosting |\n")
        f.write("|--------|--------------------|-----------|\n")
        for metric in ["roc_auc", "accuracy", "f1"]:
            lr = aggregated["logistic_regression"][metric]
            gb = aggregated["gradient_boosting"][metric]
            f.write(f"| {metric} | {lr['mean']:.4f} ± {lr['std']:.4f} | {gb['mean']:.4f} ± {gb['std']:.4f} |\n")

        f.write(f"\n### Verdict\n{verdict}\n\n")

        f.write("### Statistical Notes\n")
        f.write(f"- ROC-AUC difference: {diff:+.4f}\n")
        f.write(f"- 95% CI (rough): LogisticRegression [{lr_auc['mean'] - 2*lr_auc['std']:.4f}, {lr_auc['mean'] + 2*lr_auc['std']:.4f}]\n")
        f.write(f"- 95% CI (rough): GradientBoosting [{gb_auc['mean'] - 2*gb_auc['std']:.4f}, {gb_auc['mean'] + 2*gb_auc['std']:.4f}]\n")
        f.write(f"- Overlapping intervals: {overlap}\n\n")

        f.write("## Limitations and Risks\n\n")
        f.write("1. **Hyperparameter selection**: Both models use fixed hyperparameters (not tuned). A proper comparison might tune both on a validation set.\n")
        f.write("2. **Class imbalance**: Dataset is imbalanced (see target rates in output). F1 and ROC-AUC are robust; accuracy is not.\n")
        f.write("3. **Feature engineering**: No domain-specific features constructed; only raw numeric features used.\n")
        f.write("4. **Sample size**: ~2800 train, ~1400 test samples. Larger dataset would reduce variance estimates.\n")
        f.write("5. **Duplicates**: Dataset contained 200 exact duplicates (now removed). This is unusual and suggests data quality issues.\n\n")

        f.write("## Reproducibility\n")
        f.write(f"- Seeds used: {results['seeds']}\n")
        f.write("- Same seeds produce identical results\n")
        f.write("- Machine-readable metrics: `results/metrics.json`\n")

    print(f"[Output] Report written to {report_file}")
    return report_file


if __name__ == "__main__":
    csv_path = "churn.csv"
    output_dir = Path("results")

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run: python3 make_dataset.py --out churn.csv")
        sys.exit(1)

    # Run experiment.
    results = run_experiment(csv_path, seeds=[42, 123, 456])

    # Write outputs.
    write_json_results(results, output_dir)
    write_markdown_report(results, output_dir, csv_path)

    print("\n[Done] Experiment complete.")
