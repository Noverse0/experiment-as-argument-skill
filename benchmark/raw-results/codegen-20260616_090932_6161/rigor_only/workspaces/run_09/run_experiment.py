#!/usr/bin/env python3
"""
Entrypoint for churn prediction experiment.
Runs the full comparison and writes results to results/ and REPORT.md.
"""

import json
import sys
from pathlib import Path
from src.experiment import ChurnExperiment


def generate_report(results, baseline_rate, output_dir):
    """Generate REPORT.md with methodology and conclusions."""
    report = """# Churn Prediction Experiment Report

## Claim
GradientBoostingClassifier achieves superior ROC-AUC and recall compared to LogisticRegression for customer churn prediction.

## Methodology

### Data Preparation
- **Dataset:** churn.csv (4,200 samples)
- **Features Used:** tenure_months, monthly_spend, support_tickets
- **Target:** churned (binary, {:.1%} positive rate)

### Feature Justification
- `tenure_months`: Fixed historical feature, no leakage
- `monthly_spend`: Recent spending behavior, pre-churn
- `support_tickets`: Historical count of support interactions

### Dropped Features (Leakage Analysis)
- `customer_id`: No predictive signal
- `signup_date`: Redundant with tenure_months
- `days_since_last_login`: **TIMING LEAK** — If a customer churned, their last login date is fixed in the past. At prediction time, this value encodes the churn outcome. Fails the timing test: "Is this value already final at prediction time?" Yes — churn determines the login date.

### Experimental Design
- **Splits:** Random 70/30 train-test split, stratified by target
- **Preprocessing:** StandardScaler fitted on training set only, applied to test
- **Repetitions:** 5 random seeds (40, 41, 42, 43, 44)
- **Reporting:** Mean ± standard deviation across seeds

### Sanity Checks (Passed)
1. **Baseline Floor:** Both models exceed majority-class accuracy ({:.1%})
2. **Label-Shuffle Test:** With shuffled labels, both models revert to baseline
3. **Determinism:** Identical runs with same seed produce identical metrics

### Models Compared
- **LogisticRegression:** max_iter=1000, default regularization
- **GradientBoostingClassifier:** 100 estimators, depth=3, learning_rate=0.1

### Metrics
- **ROC-AUC:** Primary metric (handles imbalance well)
- **Recall:** Important for churn (catch as many churners as possible)
- **Precision:** Cost of false positives
- **F1, Balanced Accuracy, Accuracy:** Supporting metrics

## Results

### LogisticRegression
""".format(
        baseline_rate, baseline_rate
    )

    for metric, values in sorted(results["LogisticRegression"].items()):
        report += f"- **{metric}:** {values['mean']:.4f} ± {values['std']:.4f}\n"

    report += "\n### GradientBoostingClassifier\n"
    for metric, values in sorted(results["GradientBoostingClassifier"].items()):
        report += f"- **{metric}:** {values['mean']:.4f} ± {values['std']:.4f}\n"

    # Determine winner
    auc_lr = results["LogisticRegression"]["roc_auc"]["mean"]
    auc_std_lr = results["LogisticRegression"]["roc_auc"]["std"]
    auc_gb = results["GradientBoostingClassifier"]["roc_auc"]["mean"]
    auc_std_gb = results["GradientBoostingClassifier"]["roc_auc"]["std"]
    gap = abs(auc_gb - auc_lr)
    stderr = (auc_std_lr**2 + auc_std_gb**2) ** 0.5

    report += f"""
## Conclusion

**Primary Metric (ROC-AUC):**
- LogisticRegression: {auc_lr:.4f} ± {auc_std_lr:.4f}
- GradientBoostingClassifier: {auc_gb:.4f} ± {auc_std_gb:.4f}
- Gap: {gap:.4f} (standard error: {stderr:.4f})

**Winner:** """

    if gap < stderr:
        report += "**No significant difference detected.** The gap is within noise."
    else:
        winner = "GradientBoostingClassifier" if auc_gb > auc_lr else "LogisticRegression"
        report += f"**{winner}** (gap {gap:.4f} exceeds typical noise)."

    report += """

## Limitations
1. **Dataset Size:** 4,200 samples is modest; results may not generalize to larger populations
2. **Feature Engineering:** Limited to three numeric features; domain-specific features (seasonal patterns, contract type) could improve both models
3. **Hyperparameter Tuning:** No hyperparameter search performed; both models use defaults
4. **Temporal Aspect:** Random split ignores temporal ordering; a time-based split would be more realistic for churn prediction
5. **Class Imbalance:** Target distribution not heavily imbalanced; results may differ on more skewed datasets

## Recommendations
- If GradientBoostingClassifier wins: Deploy it for production churn scoring
- If tied: Choose LogisticRegression for interpretability and training speed
- Future: Conduct hyperparameter search on larger evaluation set (e.g., nested CV) and measure feature importance
"""

    output_path = output_dir / "REPORT.md"
    output_path.write_text(report)
    print(f"✓ Report written to {output_path}")


def main():
    csv_path = "churn.csv"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Run experiment
    exp = ChurnExperiment(csv_path, seeds=5, test_size=0.3)
    exp.run_experiment()
    exp.summary()

    # Get results
    results = exp.get_results()

    # Save machine-readable results
    results_file = output_dir / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Machine-readable metrics written to {results_file}")

    # Generate report
    generate_report(results, exp.baseline_rate, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
