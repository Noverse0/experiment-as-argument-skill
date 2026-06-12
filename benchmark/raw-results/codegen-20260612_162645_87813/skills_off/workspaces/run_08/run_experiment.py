"""Entrypoint: run the full churn experiment and write results/ + REPORT.md.

Usage:
    python3 make_dataset.py --out churn.csv   # once, to generate the data
    python3 run_experiment.py                 # runs everything (CPU, < 5 min)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment import render_report, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="churn.csv", help="path to the churn CSV")
    parser.add_argument("--results", default="results", help="results output dir")
    parser.add_argument("--report", default="REPORT.md", help="report output path")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"dataset '{args.data}' not found. Run: python3 make_dataset.py --out {args.data}"
        )

    metrics = run(args.data, args.results)
    render_report(metrics, args.report)

    concl = metrics["conclusion"]
    print("=== Sanity checks ===")
    for c in metrics["sanity_checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c.get('mean_roc_auc', c.get('train_roc_auc'))}")
    print("=== Results (ROC-AUC mean ± sd, time-series CV) ===")
    for name, arm in metrics["arms"].items():
        print(f"  {name}: {arm['roc_auc_mean']:.4f} ± {arm['roc_auc_sd']:.4f}")
    print(f"=== Conclusion: {concl['verdict']} ===")
    print(f"  {concl['statement']}")
    print(f"\nWrote {args.results}/metrics.json, {args.results}/summary.csv, {args.report}")


if __name__ == "__main__":
    main()
