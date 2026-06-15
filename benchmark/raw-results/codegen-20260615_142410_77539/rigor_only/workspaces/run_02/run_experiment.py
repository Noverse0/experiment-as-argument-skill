#!/usr/bin/env python3
"""Entry point: generate dataset and run full experiment."""
import subprocess
import sys
from src.experiment import ExperimentRunner


def generate_dataset(csv_path: str = "churn.csv") -> None:
    """Generate dataset by running make_dataset.py."""
    print(f"Generating dataset: {csv_path}")
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", csv_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error generating dataset:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())


def main():
    """Generate dataset and run experiment."""
    csv_path = "churn.csv"

    # Step 1: Generate dataset
    generate_dataset(csv_path)

    # Step 2: Run experiment
    runner = ExperimentRunner(csv_path=csv_path)
    result = runner.run_experiment(n_seeds=5)

    # Step 3: Save results
    runner.save_results(output_dir="results", experiment_result=result)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("Outputs:")
    print("  - results/metrics.json (machine-readable metrics)")
    print("  - REPORT.md (human-readable report)")


if __name__ == "__main__":
    main()
