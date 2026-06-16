#!/usr/bin/env python3
"""Entrypoint: Generate dataset, run experiment, save results and report."""
import os
import subprocess
import sys
from pathlib import Path

from src.experiment import run_experiment_multiple_seeds, save_results, generate_report


def main():
    # Create output directories
    os.makedirs("results", exist_ok=True)

    # Step 1: Generate dataset
    print("=" * 60)
    print("STEP 1: Generating dataset...")
    print("=" * 60)
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR generating dataset:", result.stderr)
        return 1

    if not os.path.exists("churn.csv"):
        print("ERROR: churn.csv not created")
        return 1

    # Step 2: Run experiment
    print("\n" + "=" * 60)
    print("STEP 2: Running experiment with multiple seeds...")
    print("=" * 60)
    try:
        results = run_experiment_multiple_seeds("churn.csv", seeds=[7, 42, 123])
    except Exception as e:
        print(f"ERROR running experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Save results
    print("\n" + "=" * 60)
    print("STEP 3: Saving results...")
    print("=" * 60)
    save_results(results, "results")

    # Step 4: Generate report
    print("\n" + "=" * 60)
    print("STEP 4: Generating report...")
    print("=" * 60)
    generate_report(results, "REPORT.md")

    print("\n" + "=" * 60)
    print("✓ Experiment complete!")
    print("=" * 60)
    print(f"Results: results/metrics.json")
    print(f"Report:  REPORT.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
