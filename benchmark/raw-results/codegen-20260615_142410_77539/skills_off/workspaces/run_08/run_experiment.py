#!/usr/bin/env python3
"""Entrypoint: Run the full churn prediction experiment."""
import os
import sys

from src.experiment import run_experiment, save_results, format_report


def main():
    csv_path = "churn.csv"
    output_dir = "results"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run: python3 make_dataset.py --out {csv_path}")
        sys.exit(1)

    # Run experiment
    results = run_experiment(csv_path, output_dir)

    # Save machine-readable results
    save_results(results, output_dir)

    # Generate and save report
    report = format_report(results)
    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    agg = results["aggregated"]
    lr_auc = agg["logistic_regression"]["test_auc"]["mean"]
    gb_auc = agg["gradient_boosting"]["test_auc"]["mean"]
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    print(f"Gradient Boosting AUC: {gb_auc:.4f}")
    print(f"Difference: {gb_auc - lr_auc:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
