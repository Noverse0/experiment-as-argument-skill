#!/usr/bin/env python3
"""
Entrypoint for the churn prediction experiment.

Usage:
  python3 run_experiment.py [--csv churn.csv] [--output results]
"""
import argparse
import sys
from pathlib import Path

# Add src to path so we can import experiment
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run churn prediction experiment (LogisticRegression vs GradientBoosting)"
    )
    parser.add_argument("--csv", default="churn.csv", help="Path to input CSV file")
    parser.add_argument(
        "--output", default="results", help="Path to output directory for results"
    )
    args = parser.parse_args()

    # Verify input file exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    run_experiment(csv_path=args.csv, output_dir=args.output)
    print("\n✓ Experiment complete. Check results/ for metrics.json and REPORT.md")


if __name__ == "__main__":
    main()
