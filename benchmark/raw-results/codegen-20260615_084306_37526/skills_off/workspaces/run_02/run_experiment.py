"""Entrypoint: run the churn prediction experiment end-to-end.

Usage:
    python3 run_experiment.py [--data churn.csv]

Outputs:
    results/metrics.json  — machine-readable per-model and comparison metrics
    REPORT.md             — methodology, results, and conclusion
"""
import argparse
import json
import os
import subprocess
import sys

from src.data import load_and_prepare
from src.evaluate import compute_sanity, run_comparison
from src.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    args = parser.parse_args()

    # Generate dataset if absent
    if not os.path.exists(args.data):
        print(f"Generating {args.data} ...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", args.data],
            check=True,
        )

    os.makedirs("results", exist_ok=True)

    print("Loading and preparing data ...")
    X_train, X_test, y_train, y_test, scaler, meta = load_and_prepare(args.data)

    print(
        f"  Original rows: {meta['original_size']}  "
        f"After dedup: {meta['deduped_size']} "
        f"(-{meta['duplicates_removed']} duplicates)"
    )
    print(
        f"  Train: {meta['train_size']} rows, churn rate {meta['train_churn_rate']:.3f}  |  "
        f"Test: {meta['test_size']} rows, churn rate {meta['test_churn_rate']:.3f}"
    )

    print("Running sanity checks ...")
    sanity = compute_sanity(X_train, X_test, y_train, y_test)
    shuffle_status = "PASS" if sanity["shuffle_test_passes"] else "FAIL"
    print(
        f"  Baseline AUC (majority class): {sanity['baseline_auc']:.4f}  |  "
        f"Label-shuffle AUC: {sanity['label_shuffle_auc']:.4f} [{shuffle_status}]"
    )

    print("Running model comparison (5 seeds × 2 models) ...")
    results = run_comparison(X_train, X_test, y_train, y_test)
    results["sanity"] = sanity
    results["data_meta"] = meta

    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics written → {metrics_path}")

    write_report(results)
    print("  Report written → REPORT.md")

    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    cmp = results["comparison"]

    print("\n--- Summary ---")
    print(
        f"  Logistic Regression  AUC {lr['auc_mean']:.4f} ± {lr['auc_std']:.4f}  "
        f"F1 {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f}"
    )
    print(
        f"  Gradient Boosting    AUC {gb['auc_mean']:.4f} ± {gb['auc_std']:.4f}  "
        f"F1 {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f}"
    )
    print(
        f"  AUC gap (GB-LR): {cmp['auc_gap']:+.4f}  |  "
        f"Conclusion: {cmp['conclusion']}"
    )


if __name__ == "__main__":
    main()
