"""Entrypoint: generate dataset, run experiment, write results and report."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.experiment import run
from src.report import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report", default="REPORT.md")
    parser.add_argument("--regen-data", action="store_true",
                        help="Re-run make_dataset.py before the experiment")
    args = parser.parse_args()

    if args.regen_data or not Path(args.data).exists():
        print(f"[main] generating {args.data} ...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", args.data],
            check=True,
        )

    summary = run(data_path=args.data, results_dir=args.results_dir)
    write_report(summary, report_path=args.report)

    print("\n=== Done ===")
    print(f"  metrics : {args.results_dir}/metrics.json")
    print(f"  report  : {args.report}")


if __name__ == "__main__":
    main()
