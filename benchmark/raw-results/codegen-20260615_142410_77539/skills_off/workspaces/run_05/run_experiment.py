#!/usr/bin/env python3
"""Entrypoint for churn prediction experiment.

Claim: Gradient boosting outperforms logistic regression on honest features
when target leaks (days_since_last_login) are excluded.
"""
import json
import os
from pathlib import Path
import sys

from src.data import (
    load_and_validate, deduplicate_before_split, split_train_test,
    preprocess, get_class_distribution
)
from src.experiment import (
    sanity_check_overfit_one_batch, sanity_check_label_shuffle,
    run_single_experiment, aggregate_results, summarize_results
)


def main():
    # Setup
    csv_path = "churn.csv"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run: python3 make_dataset.py --out churn.csv")
        sys.exit(1)

    print("=" * 60)
    print("CHURN PREDICTION EXPERIMENT")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading and validating data...")
    df = load_and_validate(csv_path)
    print(f"  Loaded {len(df)} rows")

    # 2. Deduplication
    print("\n[2/6] Deduplication...")
    df_dedup, n_removed = deduplicate_before_split(df)
    print(f"  Removed {n_removed} exact duplicates")
    print(f"  Working with {len(df_dedup)} unique rows")

    # 3. Data summary
    print("\n[3/6] Data summary:")
    class_dist = get_class_distribution(df_dedup["churned"])
    print(f"  Target distribution:")
    print(f"    - Class 0 (not churned): {class_dist['class_0_count']} ({(1 - class_dist['class_1_rate'])*100:.1f}%)")
    print(f"    - Class 1 (churned): {class_dist['class_1_count']} ({class_dist['class_1_rate']*100:.1f}%)")
    print(f"  Features used: tenure_months, monthly_spend, support_tickets")
    print(f"  WARNING: days_since_last_login EXCLUDED (post-outcome target leak)")

    # 4. Sanity checks (on full dataset splits)
    print("\n[4/6] Sanity checks...")
    X_full = df_dedup[["tenure_months", "monthly_spend", "support_tickets"]].copy()
    y_full = df_dedup["churned"].copy()
    X_train_sanity, X_test_sanity, y_train_sanity, y_test_sanity = split_train_test(
        df_dedup, test_size=0.2, random_state=42
    )
    X_train_sanity, X_test_sanity, _ = preprocess(X_train_sanity, X_test_sanity, use_scaling=True)

    check1 = sanity_check_overfit_one_batch(X_train_sanity, y_train_sanity)
    print(f"  Overfit one batch: {check1['passed']}")

    check2 = sanity_check_label_shuffle(X_train_sanity, X_test_sanity, y_train_sanity, y_test_sanity)
    print(f"  Label shuffle test: {check2['passed']}")

    sanity_checks = [check1, check2]

    # 5. Run experiments with multiple seeds
    print("\n[5/6] Running experiments with 3 seeds...")
    all_results = []
    seeds = [42, 123, 999]

    for i, seed in enumerate(seeds, 1):
        print(f"  Seed {i}/3: {seed}...")
        X_train, X_test, y_train, y_test = split_train_test(df_dedup, test_size=0.2, random_state=seed)
        X_train_scaled, X_test_scaled, _ = preprocess(X_train, X_test, use_scaling=True)

        seed_results = run_single_experiment(X_train_scaled, X_test_scaled, y_train, y_test, seed=seed)
        for result in seed_results:
            result["seed"] = seed
        all_results.extend(seed_results)

    df_results = aggregate_results(all_results)
    summary = summarize_results(df_results)

    # 6. Save results
    print("\n[6/6] Saving results...")

    # Machine-readable results
    results_dict = {
        "claim": "Gradient boosting outperforms logistic regression on honest features (target leaks excluded)",
        "data": {
            "csv": csv_path,
            "total_rows": len(df),
            "duplicates_removed": n_removed,
            "final_rows": len(df_dedup),
            "target_rate": class_dist["class_1_rate"],
        },
        "features_used": ["tenure_months", "monthly_spend", "support_tickets"],
        "features_excluded": ["days_since_last_login (post-outcome leak)"],
        "splits": 3,
        "seeds": seeds,
        "sanity_checks": sanity_checks,
        "summary": summary,
        "raw_results": all_results,
    }

    results_json = results_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Wrote {results_json}")

    # Generate report
    report = generate_report(summary, class_dist, sanity_checks, results_dict)
    report_path = Path("REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Wrote {report_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


def generate_report(summary, class_dist, sanity_checks, results_dict):
    """Generate markdown report."""
    report = []
    report.append("# Churn Prediction Experiment Report\n")

    report.append("## Claim\n")
    report.append("Gradient boosting outperforms logistic regression for predicting customer churn ")
    report.append("when target leaks are excluded and honest features are used.\n\n")

    report.append("## Design\n")
    report.append("- **Features**: tenure_months, monthly_spend, support_tickets (honest signals only)\n")
    report.append("- **Excluded**: days_since_last_login (detected as post-outcome target leak)\n")
    report.append("- **Splits**: Stratified train/test (80/20) across 3 seeds\n")
    report.append("- **Data contact**: Deduplication before split, scaler fitted on train only\n")
    report.append("- **Metrics**: AUC-ROC, precision, recall, F1, accuracy\n\n")

    report.append("## Data Summary\n")
    report.append(f"- **Total rows**: {results_dict['data']['total_rows']}\n")
    report.append(f"- **Duplicates removed**: {results_dict['data']['duplicates_removed']}\n")
    report.append(f"- **Final rows**: {results_dict['data']['final_rows']}\n")
    report.append(f"- **Target rate**: {class_dist['class_1_rate']:.1%}\n\n")

    report.append("## Sanity Checks\n")
    for check in sanity_checks:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        report.append(f"- {check['sanity_check']}: {status}\n")
    report.append("\n")

    report.append("## Results\n")
    report.append("### Mean ± Std across 3 seeds:\n\n")
    report.append("| Model | AUC-ROC | Precision | Recall | F1 | Accuracy |\n")
    report.append("|-------|---------|-----------|--------|----|-----------|\n")

    for model in ["baseline_majority", "logistic_regression", "gradient_boosting"]:
        if model in summary:
            s = summary[model]
            auc = f"{s['auc_roc'][0]:.3f} ± {s['auc_roc'][1]:.3f}"
            prec = f"{s['precision'][0]:.3f} ± {s['precision'][1]:.3f}"
            recall = f"{s['recall'][0]:.3f} ± {s['recall'][1]:.3f}"
            f1 = f"{s['f1'][0]:.3f} ± {s['f1'][1]:.3f}"
            acc = f"{s['accuracy'][0]:.3f} ± {s['accuracy'][1]:.3f}"
            report.append(f"| {model.replace('_', ' ').title()} | {auc} | {prec} | {recall} | {f1} | {acc} |\n")

    report.append("\n")

    # Interpretation
    gb_auc = summary["gradient_boosting"]["auc_roc"][0]
    lr_auc = summary["logistic_regression"]["auc_roc"][0]
    gb_std = summary["gradient_boosting"]["auc_roc"][1]
    lr_std = summary["logistic_regression"]["auc_roc"][1]

    report.append("## Interpretation\n")
    report.append(f"- Gradient Boosting AUC: {gb_auc:.3f} ± {gb_std:.3f}\n")
    report.append(f"- Logistic Regression AUC: {lr_auc:.3f} ± {lr_std:.3f}\n")
    report.append(f"- Difference: {gb_auc - lr_auc:+.3f}\n\n")

    if gb_auc > lr_auc and (gb_auc - lr_auc) > (gb_std + lr_std):
        conclusion = "**Gradient boosting outperforms logistic regression** with non-overlapping confidence intervals."
    elif gb_auc > lr_auc:
        conclusion = "Gradient boosting is slightly better, but difference is within noise (overlapping intervals)."
    else:
        conclusion = "No clear winner; logistic regression is competitive or slightly better."

    report.append(conclusion + "\n\n")

    report.append("## Methodology Notes\n")
    report.append("- **Leak detection**: days_since_last_login excluded because churned customers have ")
    report.append("systematically higher values (post-outcome). This feature would produce suspiciously high metrics.\n")
    report.append("- **Deduplication**: 200 exact duplicates found and removed before train/test split.\n")
    report.append("- **Stratified split**: Used to preserve class balance across train/test.\n")
    report.append("- **Repetition**: 3 seeds to estimate uncertainty; single seed would be anecdotal.\n\n")

    report.append("## Limitations\n")
    report.append("- Small dataset (4000 rows); results may not generalize to larger cohorts.\n")
    report.append("- Features are limited; additional features might change the conclusion.\n")
    report.append("- No hyperparameter tuning (grid search); fixed hyperparams used across both models.\n")
    report.append("- Time-based leakage not fully addressed (signup_date temporal column unused, ")
    report.append("but experiment does not claim forward-looking performance).\n")

    return "".join(report)


if __name__ == "__main__":
    main()
