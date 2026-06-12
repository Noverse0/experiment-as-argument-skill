"""
Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn prediction.

Usage:
    python3 run_experiment.py [--dataset churn.csv]
"""
import argparse
import json
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data import load_and_clean, get_X_y
from src.evaluate import evaluate_model, summarize
from src.sanity import check_baseline_floor, check_label_shuffle, check_overfit_tiny
from src.report import build_report

RESULTS_DIR = "results"
REPORT_FILE = "REPORT.md"


def main(dataset: str = "churn.csv") -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Data ----------------------------------------------------------
    df = load_and_clean(dataset)
    X, y = get_X_y(df)
    churn_rate = y.mean()
    print(f"[data] {len(df)} rows, {X.shape[1]} features, churn rate={churn_rate:.3f}")

    # ---- Sanity checks -------------------------------------------------
    print("\n[sanity] Running pipeline sanity checks...")
    probe = LogisticRegression(max_iter=1000, random_state=42)

    floor_auc = check_baseline_floor(probe, X, y)
    print(f"  baseline floor:   AUC={floor_auc:.4f}  PASS")

    shuffle_auc = check_label_shuffle(probe, X, y)
    print(f"  label shuffle:    AUC={shuffle_auc:.4f}  PASS")

    tiny_acc = check_overfit_tiny(X, y)
    print(f"  overfit tiny:     train_acc={tiny_acc:.4f}  PASS")

    # ---- Models --------------------------------------------------------
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    all_records = {}
    summaries = {}

    for name, model in models.items():
        print(f"\n[eval] {name} (5-fold × 3 seeds = 15 folds)...")
        records = evaluate_model(model, X, y)
        s = summarize(records)
        all_records[name] = records
        summaries[name] = s
        print(f"  ROC-AUC:       {s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}")
        print(f"  F1:            {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")
        print(f"  Avg-Precision: {s['avg_precision_mean']:.4f} ± {s['avg_precision_std']:.4f}")

    # ---- Persist results -----------------------------------------------
    for name, records in all_records.items():
        pd.DataFrame(records).to_csv(f"{RESULTS_DIR}/{name}_folds.csv", index=False)

    with open(f"{RESULTS_DIR}/summaries.json", "w") as fh:
        json.dump(summaries, fh, indent=2)

    sanity_meta = {
        "baseline_floor_auc": floor_auc,
        "label_shuffle_auc": shuffle_auc,
        "overfit_tiny_acc": tiny_acc,
    }
    with open(f"{RESULTS_DIR}/sanity.json", "w") as fh:
        json.dump(sanity_meta, fh, indent=2)

    print(f"\n[results] Written to {RESULTS_DIR}/")

    # ---- Report --------------------------------------------------------
    report_text = build_report(summaries, len(df), X.shape[1], churn_rate)
    with open(REPORT_FILE, "w") as fh:
        fh.write(report_text)
    print(f"[report] Written to {REPORT_FILE}")

    # Print conclusion
    auc_diff = summaries["gradient_boosting"]["roc_auc_mean"] - summaries["logistic_regression"]["roc_auc_mean"]
    print(f"\n[conclusion] ROC-AUC difference (GB − LR) = {auc_diff:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="churn.csv")
    args = parser.parse_args()
    main(args.dataset)
