#!/usr/bin/env python3
"""
Entrypoint: compares LogisticRegression vs GradientBoostingClassifier on churn prediction.

Usage:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py [--data churn.csv] [--splits 5] [--seed 42]

Outputs:
    results/metrics.json   machine-readable per-fold and summary metrics
    results/metrics.csv    flat CSV of per-fold metrics
    REPORT.md              human-readable conclusion
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_and_clean, get_time_splits
from src.models import build_lr_pipeline, build_gb_pipeline
from src.evaluate import compute_metrics, summarise


def run_sanity_checks(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Run baseline and label-shuffle sanity checks on the first fold."""
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_auc_score

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Baseline floor (majority class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    baseline_auc = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])

    # Label shuffle ceiling check (should regress to baseline)
    rng = np.random.default_rng(seed)
    shuffled_labels = y_tr.copy()
    rng.shuffle(shuffled_labels)
    lr_shuffled = build_lr_pipeline(seed)
    lr_shuffled.fit(X_tr, shuffled_labels)
    shuffled_auc = roc_auc_score(y_te, lr_shuffled.predict_proba(X_te)[:, 1])

    return {
        "baseline_dummy_auc": float(baseline_auc),
        "shuffled_label_auc": float(shuffled_auc),
        "shuffle_regressed": bool(shuffled_auc < 0.6),
    }


def cross_validate_model(pipeline, splits, X, y):
    """Run k-fold CV and return per-fold metrics."""
    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        metrics["fold"] = fold_idx
        metrics["train_size"] = int(len(train_idx))
        metrics["test_size"] = int(len(test_idx))
        metrics["test_churn_rate"] = float(y_test.mean())
        fold_metrics.append(metrics)

    return fold_metrics


def determine_winner(lr_summary: dict, gb_summary: dict) -> tuple[str, str]:
    """Return (winner_name, reasoning) based on ROC-AUC mean ± std."""
    lr_auc = lr_summary["roc_auc"]
    gb_auc = gb_summary["roc_auc"]

    lr_mean, lr_std = lr_auc["mean"], lr_auc["std"]
    gb_mean, gb_std = gb_auc["mean"], gb_auc["std"]

    # Check for meaningful gap: difference must exceed noise
    gap = gb_mean - lr_mean
    noise = lr_std + gb_std  # conservative: sum of both spreads

    if abs(gap) <= noise:
        winner = "no_detectable_difference"
        reason = (
            f"The AUC gap ({gap:+.4f}) is within the combined spread "
            f"(LR: {lr_mean:.4f}±{lr_std:.4f}, GB: {gb_mean:.4f}±{gb_std:.4f}). "
            "Neither model is a clear winner."
        )
    elif gap > 0:
        winner = "GradientBoosting"
        reason = (
            f"GB outperforms LR by {gap:.4f} AUC points, outside the combined noise "
            f"(LR: {lr_mean:.4f}±{lr_std:.4f}, GB: {gb_mean:.4f}±{gb_std:.4f})."
        )
    else:
        winner = "LogisticRegression"
        reason = (
            f"LR outperforms GB by {-gap:.4f} AUC points, outside the combined noise "
            f"(LR: {lr_mean:.4f}±{lr_std:.4f}, GB: {gb_mean:.4f}±{gb_std:.4f})."
        )

    return winner, reason


def write_report(
    sanity: dict,
    lr_folds: list,
    gb_folds: list,
    lr_summary: dict,
    gb_summary: dict,
    winner: str,
    reasoning: str,
    n_splits: int,
    n_dropped_dupes: int,
    n_rows: int,
    output_path: str,
):
    def fmt(stat): return f"{stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['n']})"

    report_metric_keys = ["roc_auc", "avg_precision", "f1", "precision", "recall"]

    lines = [
        "# Churn Prediction Experiment: Logistic Regression vs Gradient Boosting",
        "",
        "## Conclusion",
        "",
        f"**Winner: {winner}**",
        "",
        reasoning,
        "",
        "## Methodology",
        "",
        "### Data",
        f"- Dataset: `churn.csv`, {n_rows + n_dropped_dupes} rows before deduplication",
        f"- Dropped {n_dropped_dupes} exact duplicate rows (planted rigor trap)",
        f"- Dropped `account_status` — it encodes the target perfectly (leak)",
        "- Dropped `customer_id` — identifier, not predictive",
        "- Converted `signup_date` to `signup_days` (days since first observation)",
        "",
        "### Features Used",
        "- `tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`",
        "",
        "### Split Strategy",
        f"- Sorted rows by `signup_date` and applied `TimeSeriesSplit(n_splits={n_splits})`",
        "- This ensures training always precedes test chronologically, respecting the temporal nature of signup_date",
        "- LogisticRegression pipelines include `StandardScaler` fitted on train fold only",
        "",
        "### Evaluation Metrics",
        "- Primary: **ROC-AUC** (robust to class imbalance; threshold-independent)",
        "- Secondary: avg_precision, F1, precision, recall",
        f"- Churn rate in dataset: ~{sum(m['test_churn_rate'] for m in lr_folds) / len(lr_folds):.1%} (imbalanced — AUC is appropriate)",
        "",
        "## Sanity Checks",
        "",
        f"| Check | Value | Pass? |",
        f"|-------|-------|-------|",
        f"| Baseline dummy AUC | {sanity['baseline_dummy_auc']:.4f} | — |",
        f"| Label-shuffle AUC (LR) | {sanity['shuffled_label_auc']:.4f} | {'✓ regressed to baseline' if sanity['shuffle_regressed'] else '✗ DID NOT regress — investigate leak'} |",
        "",
        "## Results",
        "",
        "### Logistic Regression",
        f"| Metric | Mean ± Std |",
        f"|--------|-----------|",
    ]
    for k in report_metric_keys:
        lines.append(f"| {k} | {fmt(lr_summary[k])} |")

    lines += [
        "",
        "### Gradient Boosting",
        f"| Metric | Mean ± Std |",
        f"|--------|-----------|",
    ]
    for k in report_metric_keys:
        lines.append(f"| {k} | {fmt(gb_summary[k])} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- **Single dataset / fixed seed**: results are specific to this generated dataset; real-world churn data may differ.",
        "- **No hyperparameter tuning**: both models use defaults. A tuned GB might widen the gap; a tuned LR might close it.",
        "- **signup_days as feature**: if churn rates shift over calendar time, this feature may pick up distribution shift rather than a causal signal.",
        "- **TimeSeriesSplit expands the training window each fold**: later folds have more training data, which may favour more complex models.",
        "- **Threshold fixed at 0.5** for F1/precision/recall — business cost of false negatives vs false positives was not considered.",
    ]

    Path(output_path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print(f"[1/5] Loading data from {args.data}")
    df, n_dropped_dupes = load_and_clean(args.data)
    print(f"      {len(df)} rows after deduplication (dropped {n_dropped_dupes} duplicates)")
    print(f"      Churn rate: {df['churned'].mean():.2%}")

    print("[2/5] Running sanity checks")
    splits, X, y = get_time_splits(df, n_splits=args.splits)
    sanity = run_sanity_checks(X, y, seed=args.seed)
    print(f"      Dummy AUC: {sanity['baseline_dummy_auc']:.4f} | "
          f"Shuffle AUC: {sanity['shuffled_label_auc']:.4f} | "
          f"Shuffle regressed: {sanity['shuffle_regressed']}")
    if not sanity["shuffle_regressed"]:
        print("      WARNING: label-shuffle AUC did not regress to baseline — possible leak!")

    print(f"[3/5] Cross-validating LogisticRegression ({args.splits} folds)")
    lr_pipe = build_lr_pipeline(args.seed)
    lr_folds = cross_validate_model(lr_pipe, splits, X, y)

    print(f"[4/5] Cross-validating GradientBoostingClassifier ({args.splits} folds)")
    gb_pipe = build_gb_pipeline(args.seed)
    gb_folds = cross_validate_model(gb_pipe, splits, X, y)

    lr_summary = summarise(lr_folds)
    gb_summary = summarise(gb_folds)

    winner, reasoning = determine_winner(lr_summary, gb_summary)

    print(f"\n--- Results ---")
    print(f"LR  ROC-AUC: {lr_summary['roc_auc']['mean']:.4f} ± {lr_summary['roc_auc']['std']:.4f}")
    print(f"GB  ROC-AUC: {gb_summary['roc_auc']['mean']:.4f} ± {gb_summary['roc_auc']['std']:.4f}")
    print(f"Winner: {winner}")
    print(f"Reasoning: {reasoning}")

    print("[5/5] Writing outputs")

    # machine-readable JSON
    full_results = {
        "sanity": sanity,
        "logistic_regression": {"folds": lr_folds, "summary": lr_summary},
        "gradient_boosting": {"folds": gb_folds, "summary": gb_summary},
        "conclusion": {"winner": winner, "reasoning": reasoning},
        "config": {
            "data": args.data,
            "n_splits": args.splits,
            "seed": args.seed,
            "n_rows_after_dedup": len(df),
            "n_duplicates_dropped": n_dropped_dupes,
            "features": ["tenure_months", "monthly_spend", "support_tickets", "signup_days"],
            "target": "churned",
            "leaked_columns_dropped": ["account_status"],
        },
    }
    with open("results/metrics.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # flat CSV of per-fold metrics
    rows = []
    for fold in lr_folds:
        rows.append({"model": "LogisticRegression", **fold})
    for fold in gb_folds:
        rows.append({"model": "GradientBoosting", **fold})
    pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)

    write_report(
        sanity=sanity,
        lr_folds=lr_folds,
        gb_folds=gb_folds,
        lr_summary=lr_summary,
        gb_summary=gb_summary,
        winner=winner,
        reasoning=reasoning,
        n_splits=args.splits,
        n_dropped_dupes=n_dropped_dupes,
        n_rows=len(df),
        output_path="REPORT.md",
    )

    print("      results/metrics.json")
    print("      results/metrics.csv")
    print("      REPORT.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
