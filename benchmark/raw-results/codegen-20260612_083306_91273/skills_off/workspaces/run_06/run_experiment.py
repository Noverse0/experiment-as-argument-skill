#!/usr/bin/env python3
"""
Entrypoint for the churn prediction experiment.

Runs:
  1. Sanity checks (baseline floor, label shuffle, etc.)
  2. Full experiment with 5 seeds
  3. Writes results/results.json and REPORT.md
"""
import json
import sys
import time
from pathlib import Path

from src.experiment import load_data, preprocess_data, run_experiment
from src.sanity_checks import run_sanity_checks


def generate_report(result: dict, output_path: str = "REPORT.md"):
    """Write human-readable report from experiment results."""

    lr_auc = result['models']['LogisticRegression']['auc']
    gb_auc = result['models']['GradientBoosting']['auc']
    effect_size = result['effect_size']

    with open(output_path, 'w') as f:
        f.write("# Customer Churn Prediction: LogisticRegression vs GradientBoosting\n\n")

        f.write("## Claim\n")
        f.write("Does gradient boosting outperform logistic regression for predicting customer churn?\n\n")

        f.write("## Design\n")
        f.write("- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)\n")
        f.write("- **Fixed:** Data split (80/20 train/test, stratified), preprocessing (StandardScaler), features (tenure_months, monthly_spend, support_tickets)\n")
        f.write("- **Repetitions:** 5 random seeds\n")
        f.write("- **Metric:** ROC-AUC (robust to class imbalance), plus accuracy and PR-AUC for context\n")
        f.write("- **Test set:** Touched once at the end. All preprocessing fit on train only.\n\n")

        f.write("## Data Hygiene\n")
        f.write(f"- **Duplicates detected and removed:** {result['n_duplicates']}\n")
        f.write("- **Features dropped (justification):**\n")
        f.write("  - `account_status`: Leaked from target (\"closed\" iff churned==1)\n")
        f.write("  - `signup_date`: Temporal column; random split ignores time ordering\n")
        f.write("  - `customer_id`: Identifier, not predictive\n")
        f.write("- **Features kept:**\n")
        f.write("  - `tenure_months`: Months as customer (predictive)\n")
        f.write("  - `monthly_spend`: Monthly spending (predictive)\n")
        f.write("  - `support_tickets`: Support tickets (predictive)\n\n")

        f.write("## Results\n\n")
        f.write("### ROC-AUC (primary metric)\n")
        f.write(f"| Model | Mean | Std | Min | Max |\n")
        f.write(f"|-------|------|-----|-----|-----|\n")
        f.write(f"| LogisticRegression | {lr_auc['mean']:.4f} | {lr_auc['std']:.4f} | {lr_auc['min']:.4f} | {lr_auc['max']:.4f} |\n")
        f.write(f"| GradientBoosting | {gb_auc['mean']:.4f} | {gb_auc['std']:.4f} | {gb_auc['min']:.4f} | {gb_auc['max']:.4f} |\n\n")

        f.write("### Effect Size\n")
        f.write(f"- **Mean difference (GB - LR):** {effect_size['mean_diff_auc']:.4f}\n")
        f.write(f"- **Cohen's d:** {effect_size['cohens_d']:.4f}\n")
        f.write(f"- **Confidence intervals overlap:** {effect_size['overlapping']}\n\n")

        f.write("### Accuracy and PR-AUC (for reference)\n")
        f.write(f"| Model | Accuracy (mean ± std) | PR-AUC (mean ± std) |\n")
        f.write(f"|-------|---|---|\n")
        lr_acc = result['models']['LogisticRegression']['accuracy']
        gb_acc = result['models']['GradientBoosting']['accuracy']
        lr_pr = result['models']['LogisticRegression']['pr_auc']
        gb_pr = result['models']['GradientBoosting']['pr_auc']
        f.write(f"| LogisticRegression | {lr_acc['mean']:.4f} ± {lr_acc['std']:.4f} | {lr_pr['mean']:.4f} ± {lr_pr['std']:.4f} |\n")
        f.write(f"| GradientBoosting | {gb_acc['mean']:.4f} ± {gb_acc['std']:.4f} | {gb_pr['mean']:.4f} ± {gb_pr['std']:.4f} |\n\n")

        f.write("## Conclusion\n")
        if not effect_size['overlapping']:
            if effect_size['mean_diff_auc'] > 0:
                verdict = f"GradientBoosting **outperforms** LogisticRegression (Δ AUC = {effect_size['mean_diff_auc']:.4f}, Cohen's d = {effect_size['cohens_d']:.2f})"
            else:
                verdict = f"LogisticRegression **outperforms** GradientBoosting (Δ AUC = {-effect_size['mean_diff_auc']:.4f}, Cohen's d = {-effect_size['cohens_d']:.2f})"
        else:
            verdict = f"**No detectable difference** between models (Δ AUC = {effect_size['mean_diff_auc']:.4f} with overlapping confidence intervals)."

        f.write(f"{verdict}\n\n")

        f.write("## Limitations & Next Steps\n")
        f.write("- **Sample size:** 3,360 examples per seed (after dedup). Larger datasets could refine effect size estimates.\n")
        f.write("- **Hyperparameter tuning:** Models use defaults. A tuning budget (e.g., on a validation set) could improve both.\n")
        f.write("- **Temporal aspect:** `signup_date` was dropped. If the task is forward-looking, a time-based split should be used instead.\n")
        f.write("- **Feature engineering:** Only raw features used. Domain-driven features (e.g., spend-per-tenure ratio) might improve both models.\n")
        f.write("- **Statistical test:** For a formal p-value, a permutation test or bootstrapped CI could be used (deferred).\n\n")

        f.write("## Artifacts\n")
        f.write("- `results/results.json`: Machine-readable metrics (mean, std, min, max per model per seed)\n")
        f.write("- `results/metrics_by_seed.csv`: Raw metrics for each seed\n")
        f.write("- `REPORT.md`: This human-readable summary\n")


def main():
    print("=" * 60)
    print("Churn Prediction Experiment: LogisticRegression vs GradientBoosting")
    print("=" * 60)

    start_time = time.time()

    # Load and preprocess
    print("\n[1/4] Loading data...")
    df = load_data("churn.csv")
    df_processed = preprocess_data(df)
    print(f"  Loaded {len(df)} rows, {len(df_processed)} after preprocessing")

    # Sanity checks
    print("\n[2/4] Running sanity checks...")
    if not run_sanity_checks(df_processed):
        print("ERROR: Sanity checks failed. Do not trust results.")
        sys.exit(1)

    # Full experiment
    print("[3/4] Running full experiment (5 seeds)...")
    result = run_experiment("churn.csv", n_seeds=5, output_dir="results")

    # Report
    print("[4/4] Generating report...")
    generate_report(result)
    print(f"  Wrote REPORT.md and results/results.json")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✓ Experiment complete in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
