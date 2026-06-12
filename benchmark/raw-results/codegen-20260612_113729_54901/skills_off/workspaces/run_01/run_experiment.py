"""Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn.

Usage:
    python3 run_experiment.py [--data churn.csv]

Writes:
    results/metrics.json   — machine-readable per-seed and summary metrics
    REPORT.md              — methodology, results, and honest conclusion
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from src.data import load_and_clean, get_features_and_target, class_balance_report
from src.pipeline import build_lr_pipeline, build_gb_pipeline
from src.evaluate import evaluate_pipeline, label_shuffle_check
from src.report import write_report

SEEDS = [42, 123, 777]
N_SPLITS = 5


def _baseline_auc(X, y) -> float:
    """Majority-class dummy baseline — models must beat this."""
    split = int(len(X) * 0.8)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X.iloc[:split], y.iloc[:split])
    proba = dummy.predict_proba(X.iloc[split:])[:, 1]
    return float(roc_auc_score(y.iloc[split:], proba))


def main(data_path: str = "churn.csv") -> None:
    Path("results").mkdir(exist_ok=True)

    # ── Load and clean ────────────────────────────────────────────────────────
    n_before_dedup = sum(1 for _ in open(data_path)) - 1  # subtract header
    df = load_and_clean(data_path)
    n_dups_removed = n_before_dedup - len(df)

    X, y = get_features_and_target(df)
    balance = class_balance_report(y)

    print(f"\nDataset: {balance['n_samples']} rows | churn rate: {balance['positive_rate']:.1%}")
    print(f"Features: {list(X.columns)}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n── Sanity checks ──────────────────────────────────────────────────")
    baseline_auc = _baseline_auc(X, y)
    print(f"  Majority-class baseline ROC-AUC: {baseline_auc:.3f} (models must exceed this)")

    lr_shuffle = label_shuffle_check(build_lr_pipeline(), X, y, seed=42)
    gb_shuffle = label_shuffle_check(build_gb_pipeline(), X, y, seed=42)
    print(f"  Label-shuffle ROC-AUC — LR: {lr_shuffle:.3f}  GB: {gb_shuffle:.3f}  (expect ~0.5)")
    if lr_shuffle > 0.6 or gb_shuffle > 0.6:
        print("  WARNING: label-shuffle AUC is suspiciously high — audit features for leakage.")

    sanity = {
        "baseline_auc": baseline_auc,
        "lr_shuffle_auc": lr_shuffle,
        "gb_shuffle_auc": gb_shuffle,
    }

    # ── Cross-validated evaluation ────────────────────────────────────────────
    print("\n── Evaluation ─────────────────────────────────────────────────────")
    all_results = {"logistic_regression": [], "gradient_boosting": []}

    for seed in SEEDS:
        print(f"  Seed {seed} ...", end=" ", flush=True)
        lr_res = evaluate_pipeline(build_lr_pipeline(seed), X, y, N_SPLITS)
        gb_res = evaluate_pipeline(build_gb_pipeline(seed), X, y, N_SPLITS)
        all_results["logistic_regression"].append(lr_res)
        all_results["gradient_boosting"].append(gb_res)
        print(f"LR AUC={lr_res['roc_auc']['mean']:.3f}  GB AUC={gb_res['roc_auc']['mean']:.3f}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    def _aggregate(results_list: list) -> dict:
        def _agg_metric(key):
            means = [r[key]["mean"] for r in results_list]
            return {"mean": float(np.mean(means)), "std": float(np.std(means))}

        return {
            "roc_auc": _agg_metric("roc_auc"),
            "f1": _agg_metric("f1"),
            "pr_auc": _agg_metric("pr_auc"),
        }

    summary = {
        "logistic_regression": _aggregate(all_results["logistic_regression"]),
        "gradient_boosting": _aggregate(all_results["gradient_boosting"]),
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "dataset": {
            "n_rows_raw": n_before_dedup,
            "n_rows_cleaned": len(df),
            "n_dups_removed": n_dups_removed,
            "churn_rate": balance["positive_rate"],
        },
        "sanity": sanity,
    }

    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"summary": summary, "per_seed": all_results}, f, indent=2)
    print(f"\nWrote {metrics_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(
        summary=summary,
        n_rows=len(df),
        churn_rate=balance["positive_rate"],
        n_dups_removed=n_dups_removed,
        sanity=sanity,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n── Results ────────────────────────────────────────────────────────")
    lr = summary["logistic_regression"]
    gb = summary["gradient_boosting"]
    print(f"  Logistic Regression : ROC-AUC {lr['roc_auc']['mean']:.3f} ± {lr['roc_auc']['std']:.3f}")
    print(f"  Gradient Boosting   : ROC-AUC {gb['roc_auc']['mean']:.3f} ± {gb['roc_auc']['std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    args = parser.parse_args()
    main(args.data)
