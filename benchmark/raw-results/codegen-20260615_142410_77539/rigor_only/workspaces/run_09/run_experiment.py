#!/usr/bin/env python3
"""
Entrypoint: run the full churn prediction experiment.

Usage:
    python run_experiment.py

Output:
    - results/metrics.json: machine-readable metrics
    - REPORT.md: human-readable comparison report
"""

from src.experiment import ChurnExperiment


def main():
    print("=" * 70)
    print("Churn Prediction Experiment: Gradient Boosting vs Logistic Regression")
    print("=" * 70)

    experiment = ChurnExperiment(data_path="churn.csv", n_seeds=5)

    print("\n[1/3] Loading and preparing data...")
    experiment.run_experiment(drop_leakage=True)

    print("[2/3] Aggregating results...")
    agg = experiment.save_results(results_dir="results")

    print("[3/3] Generating report...")
    experiment.generate_report(agg, report_path="REPORT.md")

    print("\n" + "=" * 70)
    print("✓ Experiment complete")
    print("  - Detailed metrics: results/metrics.json")
    print("  - Report: REPORT.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
