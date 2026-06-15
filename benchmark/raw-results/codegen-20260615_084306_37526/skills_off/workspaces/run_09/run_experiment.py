#!/usr/bin/env python3
"""Entrypoint: generate data, run experiment, write results and report."""
import subprocess
import sys
from pathlib import Path

DATASET_PATH = Path("churn.csv")
RESULTS_PATH = Path("results/metrics.json")
REPORT_PATH = Path("REPORT.md")


def main() -> None:
    # Step 1: generate dataset
    print("Generating dataset...")
    subprocess.run(
        [sys.executable, "make_dataset.py", "--out", str(DATASET_PATH)],
        check=True,
    )

    # Step 2: load and prepare data
    from src.experiment import load_and_prepare, run_experiment
    from src.report import generate_report, save_results

    print("Loading and preparing data...")
    X, y = load_and_prepare(str(DATASET_PATH))
    print(f"  {len(y)} rows (after dedup), churn rate: {y.mean():.1%}")

    # Step 3: run experiment
    print("Running cross-validation (5 temporal folds × 2 models)...")
    results = run_experiment(X, y, n_splits=5)

    for name, metrics in results["models"].items():
        auc = metrics["roc_auc"]
        print(f"  {name}: ROC-AUC = {auc['mean']:.3f} ± {auc['std']:.3f}")

    # Step 4: save artifacts
    save_results(results, RESULTS_PATH)
    generate_report(results, RESULTS_PATH, REPORT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
