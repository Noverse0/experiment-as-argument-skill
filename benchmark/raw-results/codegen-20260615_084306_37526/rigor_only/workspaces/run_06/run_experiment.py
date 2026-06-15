"""Entrypoint: run the full LR vs GradientBoosting churn experiment.

Usage:
    python3 run_experiment.py [--data churn.csv]

Outputs:
    results/metrics.json   machine-readable metrics
    REPORT.md              comparison conclusion, methodology, limitations
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable when run from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from src.experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="LR vs GradientBoosting churn experiment")
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Dataset not found at {args.data!r}. Generating...")
        subprocess.run(
            ["python3", "make_dataset.py", "--out", args.data],
            check=True,
        )

    run(args.data)


if __name__ == "__main__":
    main()
