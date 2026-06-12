#!/usr/bin/env python3
"""Entrypoint: run churn prediction experiment, write results and report."""
import argparse
import sys

from src.experiment import run_experiment
from src.report import write_report


def main():
    parser = argparse.ArgumentParser(
        description="Compare LogisticRegression vs GradientBoosting on churn data."
    )
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    parser.add_argument("--results-dir", default="results", help="Output directory for metrics")
    parser.add_argument("--report", default="REPORT.md", help="Path for markdown report")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    results = run_experiment(args.data, args.results_dir)

    di = results["data_info"]
    print(
        f"  Dataset: {di['n_before_dedup']} rows, "
        f"{di['n_dupes_removed']} duplicates removed, "
        f"{di['n_train']} train / {di['n_test']} test"
    )

    sc = results["sanity_checks"]
    print(f"  Sanity — baseline AUC: {sc['baseline_majority_roc_auc']:.4f}, "
          f"label-shuffle AUC: {sc['label_shuffle_roc_auc']:.4f} "
          f"({'OK — no leakage signal' if sc['label_shuffle_near_chance'] else 'WARNING: above chance with shuffled labels'})")

    lr = results["logistic_regression"]["roc_auc"]
    gb = results["gradient_boosting"]["roc_auc"]
    print(f"  LR  ROC-AUC: {lr['mean']:.4f} ± {lr['std']:.4f}  (n={lr['n']})")
    print(f"  GB  ROC-AUC: {gb['mean']:.4f} ± {gb['std']:.4f}  (n={gb['n']})")

    write_report(results, args.report)
    print(f"\nResults written to {args.results_dir}/metrics.json")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
