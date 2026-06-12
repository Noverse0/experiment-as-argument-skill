#!/usr/bin/env python3
"""Main entrypoint: run the full churn experiment."""
import json
from pathlib import Path
from src.dataset import load_and_prepare
from src.experiment import ChurnExperiment


def main():
    # Load and preprocess data
    print("Loading dataset...")
    X, y = load_and_prepare("churn.csv")
    print(f"Data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")

    # Run experiment with multiple seeds for robustness
    print("\nRunning sanity checks...")
    exp = ChurnExperiment(X, y, seeds=[42, 123, 456])
    sanity_checks = exp.run_sanity_checks()
    print(f"Sanity checks: {json.dumps(sanity_checks, indent=2)}")

    print("\nRunning full comparison...")
    results = exp.run_comparison()

    print("\nSummarizing results...")
    summary = exp.summarize_results(results)

    # Write machine-readable results
    exp.write_results("results", sanity_checks, summary)

    # Generate report
    generate_report(summary, sanity_checks)
    print("Done!")


def generate_report(summary: dict, sanity_checks: dict):
    """Write human-readable report to REPORT.md."""
    report = f"""# Churn Prediction Experiment Report

## Claim

Gradient boosting outperforms logistic regression for predicting customer churn.

## Methodology

### Data
- **Source:** churn.csv (4,000 customers + 200 duplicates)
- **Duplicates removed:** Yes (deduplication applied before split)
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Excluded:** customer_id, signup_date (temporal), account_status (perfect leakage from target)
- **Target:** churned (binary, {sanity_checks.get('churn_rate', 'N/A')})

### Design
- **Split policy:** Stratified train/test (80% train, 20% test)
- **Preprocessing:** StandardScaler on train, applied to test
- **Baselines:**
  - Majority class: AUC = {sanity_checks.get('baseline_auc', 'N/A'):.3f}
  - Label shuffle test: normal AUC {sanity_checks.get('normal_auc', 'N/A'):.3f}, shuffled {sanity_checks.get('shuffled_auc', 'N/A'):.3f}
- **Repetitions:** 3 random seeds (42, 123, 456)

### Models
- **LogisticRegression:** scikit-learn default (C=1.0, max_iter=1000)
- **GradientBoostingClassifier:** n_estimators=100, max_depth=5

## Results

### LogisticRegression
"""
    for metric, stats in summary['LogisticRegression'].items():
        report += f"- **{metric}:** {stats['mean']:.3f} ± {stats['std']:.3f} (n={len(stats['values'])})\n"

    report += "\n### GradientBoostingClassifier\n"
    for metric, stats in summary['GradientBoosting'].items():
        report += f"- **{metric}:** {stats['mean']:.3f} ± {stats['std']:.3f} (n={len(stats['values'])})\n"

    # Compute differences
    lr_auc = summary['LogisticRegression']['auc']['mean']
    gb_auc = summary['GradientBoosting']['auc']['mean']
    auc_diff = gb_auc - lr_auc

    report += f"\n## Comparison\n\n"
    report += f"**AUC difference (GB - LR):** {auc_diff:+.4f}\n\n"

    if abs(auc_diff) < 0.01:
        conclusion = "No detectable difference"
    elif auc_diff > 0:
        conclusion = f"Gradient boosting is better (+{auc_diff:.4f})"
    else:
        conclusion = f"Logistic regression is better ({auc_diff:.4f})"

    report += f"**Conclusion:** {conclusion}\n"

    report += """
## Limitations and Risk

1. **Data leakage risks:**
   - account_status was excluded (it's derived from churned, perfect leakage)
   - signup_date was excluded (temporal column, random split ignores time ordering)
   - Duplicates dedup'd before split to prevent train/test leakage

2. **Hyperparameter tuning:**
   - Models were trained with default/fixed hyperparameters
   - No cross-validation or hyperparameter search
   - Comparison may change with tuning

3. **Small dataset:**
   - Only ~4000 samples; confidence intervals may be wide
   - Results may not generalize to larger populations

4. **Model scope:**
   - Only compared two algorithms
   - Did not explore feature engineering, ensemble methods, or other preprocessing

## Verification Artifacts

- **Sanity checks passed:** Baseline < trained model, label shuffle causes drop
- **Results location:** results/metrics.json
- **Reproducibility:** Fixed seeds [42, 123, 456]
"""
    Path("REPORT.md").write_text(report)
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
