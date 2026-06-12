#!/usr/bin/env python3
"""Entrypoint for the churn prediction experiment."""

import json
import sys
from pathlib import Path

from src.experiment import run_experiment, generate_report


def main():
    """Run experiment and generate outputs."""
    # Paths
    data_path = Path('churn.csv')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Run experiment with 5 seeds for robustness
    seeds = [42, 123, 456, 789, 999]
    print("Running experiment...")
    results = run_experiment(str(data_path), seeds, str(results_dir))

    # Write machine-readable results
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_file}")

    # Write human-readable report
    report = generate_report(results)
    report_file = Path('REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report written to {report_file}")

    # Print summary
    print("\n" + "=" * 60)
    print(results['conclusion'])
    print("=" * 60)
    print(f"\nLogistic Regression ROC-AUC:  {results['summary']['logistic_regression']['roc_auc_mean']:.4f} ± {results['summary']['logistic_regression']['roc_auc_std']:.4f}")
    print(f"Gradient Boosting ROC-AUC:    {results['summary']['gradient_boosting']['roc_auc_mean']:.4f} ± {results['summary']['gradient_boosting']['roc_auc_std']:.4f}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
