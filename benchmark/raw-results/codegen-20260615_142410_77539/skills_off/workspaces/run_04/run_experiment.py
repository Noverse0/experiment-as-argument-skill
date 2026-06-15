#!/usr/bin/env python3
"""Main entry point: run the full churn prediction experiment."""
import sys
from src.experiment import run_full_experiment, generate_report


def main():
    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT: LogisticRegression vs GradientBoosting")
    print("=" * 70)

    # Seeds for reproducibility
    seeds = [42, 123, 999]

    # Run experiment
    results = run_full_experiment("churn.csv", seeds=seeds, out_dir="results")

    # Generate report
    print("\nGenerating report...")
    generate_report(results, out_dir="results")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Results: results/metrics.json")
    print(f"Report:  results/REPORT.md")


if __name__ == "__main__":
    main()
