#!/usr/bin/env python3
"""
Entrypoint for churn prediction experiment.

Usage:
    python3 run_experiment.py

Outputs:
    - results/metrics.json: machine-readable metrics
    - REPORT.md: human-readable methodology and results
"""
import logging
import sys
from pathlib import Path

from src.experiment import run_all_seeds, summarize_results, write_results, write_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    csv_path = "churn.csv"

    # Verify dataset exists
    if not Path(csv_path).exists():
        logger.error(f"Dataset not found: {csv_path}")
        logger.info("Run: python3 make_dataset.py --out churn.csv")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Churn Prediction Experiment")
    logger.info("Comparing: LogisticRegression vs GradientBoostingClassifier")
    logger.info("=" * 60)

    # Run experiment across seeds
    seeds = [42, 123, 456, 789, 999]
    logger.info(f"Running {len(seeds)} seeds: {seeds}")
    results = run_all_seeds(csv_path, seeds)

    # Summarize
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    summary = summarize_results(results)
    for algo, stats in summary.items():
        logger.info(
            f"{algo}: "
            f"test_auc={stats['test_auc_mean']:.4f}±{stats['test_auc_std']:.4f}, "
            f"test_pr_auc={stats['test_pr_auc_mean']:.4f}±{stats['test_pr_auc_std']:.4f}"
        )

    # Write outputs
    write_results(summary, output_dir="results")
    write_report(summary, output_file="REPORT.md")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment complete!")
    logger.info("Results: results/metrics.json")
    logger.info("Report: REPORT.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
