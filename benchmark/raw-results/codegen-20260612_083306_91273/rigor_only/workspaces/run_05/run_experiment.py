#!/usr/bin/env python3
"""Entrypoint: run the full churn prediction experiment.

Usage:
    python run_experiment.py [--csv churn.csv] [--output results]
"""
import argparse
import json
import logging
from pathlib import Path

from src.experiment import run_full_experiment, summarize_results, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run rigorous churn prediction experiment: LogReg vs GradBoost"
    )
    parser.add_argument(
        "--csv", default="churn.csv", help="Path to churn dataset CSV"
    )
    parser.add_argument(
        "--output", default="results", help="Output directory for results"
    )
    args = parser.parse_args()

    logger.info("Starting churn prediction experiment")
    logger.info(f"Data: {args.csv}")
    logger.info(f"Output: {args.output}")

    # Run experiment
    results = run_full_experiment(args.csv)

    # Summarize
    summary = summarize_results(results)

    # Save
    save_results(results, summary, args.output)

    # Print summary to console
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"LogisticRegression: {summary['LogisticRegression']['test_auc_mean']:.4f} ± {summary['LogisticRegression']['test_auc_std']:.4f}")
    logger.info(f"GradientBoosting:   {summary['GradientBoosting']['test_auc_mean']:.4f} ± {summary['GradientBoosting']['test_auc_std']:.4f}")
    logger.info(f"Delta (GB - LR):    {summary['delta_auc']:.4f}")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    main()
