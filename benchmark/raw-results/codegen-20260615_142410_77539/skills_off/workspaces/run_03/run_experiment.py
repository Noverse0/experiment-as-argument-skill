#!/usr/bin/env python3
"""Entrypoint: run the churn prediction experiment."""
import sys
from src.experiment import ChurnExperiment


def main():
    """Run experiment and save results."""
    experiment = ChurnExperiment(csv_path="churn.csv", output_dir="results")
    experiment.run(seeds=[42, 43, 44])
    experiment.save_results()
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("Results saved to results/ and REPORT.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
