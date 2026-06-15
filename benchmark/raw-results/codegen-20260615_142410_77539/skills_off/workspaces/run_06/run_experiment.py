#!/usr/bin/env python3
"""Entrypoint for the churn prediction experiment."""
import sys
from src.experiment import run_experiment, write_report


def main():
    """Generate data, run experiment, write results."""
    import subprocess

    # Generate dataset if not present
    print("Generating dataset...")
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error generating dataset: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

    # Run experiment
    print("\nRunning experiment with 5 seeds...\n")
    results = run_experiment(
        data_path="churn.csv",
        seeds=[42, 123, 456, 789, 999],
        output_dir="results",
    )

    # Write report
    print("\nGenerating report...")
    write_report(results, output_path="REPORT.md")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(
        f"LR AUC:  {results['summary']['lr']['auc_mean']:.4f} ± {results['summary']['lr']['auc_std']:.4f}"
    )
    print(
        f"GB AUC:  {results['summary']['gb']['auc_mean']:.4f} ± {results['summary']['gb']['auc_std']:.4f}"
    )
    print(f"Difference (GB - LR): {results['summary']['difference']:+.4f}")
    print("\nOutputs:")
    print("  - REPORT.md (human-readable)")
    print("  - results/metrics.json (machine-readable)")


if __name__ == "__main__":
    main()
