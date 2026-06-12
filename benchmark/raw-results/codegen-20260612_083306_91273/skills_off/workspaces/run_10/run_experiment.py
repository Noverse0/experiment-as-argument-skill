#!/usr/bin/env python3
"""
Entrypoint to run the churn prediction experiment.

Usage:
  python3 run_experiment.py --data churn.csv --output results
"""
import argparse
import sys
from src.experiment import run_experiment, summarize_results


def main():
    parser = argparse.ArgumentParser(description="Run churn prediction experiment")
    parser.add_argument("--data", default="churn.csv", help="Path to churn dataset CSV")
    parser.add_argument("--output", default="results", help="Directory to write results")
    args = parser.parse_args()

    print(f"[START] Churn Prediction Experiment")
    print(f"[INPUT] Dataset: {args.data}")
    print(f"[OUTPUT] Directory: {args.output}")

    # Run experiment
    results = run_experiment(args.data, args.output)

    # Write report
    report = summarize_results(results)
    report_path = f"{args.output}/REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n[OUTPUT] Report saved to {report_path}")
    print(f"\n[COMPLETE] Experiment finished successfully\n")

    # Print summary
    print(report)


if __name__ == "__main__":
    main()
