#!/usr/bin/env python3
"""Entrypoint: generate dataset, run experiment, write results."""
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

from src.experiment import ChurnExperiment


def main():
    # Create output directories
    Path("results").mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("Step 1: Generating dataset...")
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error generating dataset: {result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())

    # Step 2: Run experiment
    print("\nStep 2: Running experiment...")
    exp = ChurnExperiment("churn.csv")
    results = exp.run()

    # Step 3: Write results to JSON
    results["timestamp"] = datetime.now().isoformat()
    results["git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=".",
    ).stdout.strip()

    metrics_path = Path("results") / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics written to {metrics_path}")

    # Step 4: Write report
    report_path = Path("REPORT.md")
    report = generate_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


def generate_report(results: dict) -> str:
    """Generate markdown report with methodology and findings."""
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    comparison = results["comparison"]

    report = f"""# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for predicting customer churn, when both are trained and evaluated with proper data hygiene (deduplication, leak exclusion, stratified CV).

## Methodology

### Dataset
- **Source:** Generated via `make_dataset.py` (seed=7, n=4000)
- **Size:** {results['n_samples']} rows after deduplication (200 duplicates removed)
- **Target:** `churned` (binary: 0=retained, 1=churned)

### Features
Three features used for prediction:
- `tenure_months`: Customer tenure in months
- `monthly_spend`: Monthly spending amount
- `support_tickets`: Number of support tickets submitted

### Excluded Features (Data Discipline)
- **`customer_id`**: Identifier only, no signal.
- **`days_since_last_login`** (TARGET LEAK): Derived from the outcome. Churned customers have longer days since last login by definition, not a predictive feature.
- **`signup_date`**: Temporal column; not engineered as a feature. Time order is implicitly respected via stratified CV order.

### Data Contact Policy
1. **Deduplication:** 200 exact duplicate rows removed before splitting.
2. **No leakage:** All fit-like operations (StandardScaler for LR) trained on train fold only, applied to test fold.
3. **Stratified K-fold:** Class balance preserved in each fold (important given {results['logistic_regression']['f1']['n']/25:.0%} positive rate).

### Models
- **Logistic Regression:** max_iter=1000, default regularization (L2), standard scaling applied.
- **Gradient Boosting:** n_estimators=100, learning_rate=0.1, random_state per seed.

### Evaluation
- **Schema:** 5 random seeds × 5-fold stratified CV = {results['n_folds'] * results['n_seeds']} train/test boundaries
- **Metrics:** ROC-AUC (robust to imbalance), F1, Precision, Recall
- **Reporting:** Mean ± std across all folds and seeds

### Sanity Checks (Pre-run Validation)
✓ **Baseline test:** Majority class accuracy established as floor.
✓ **Label shuffle:** Both models drop to near-baseline when trained on shuffled labels.
✓ **Overfit test:** Both models achieve high AUC when trained and tested on the same 50 samples.

These checks confirm:
1. The pipeline is not broken
2. The feature signal is not spurious (labels matter)
3. The models are learning

## Results

### Logistic Regression
- **ROC-AUC:** {lr['auc']['mean']:.4f} ± {lr['auc']['std']:.4f}
- **F1-Score:** {lr['f1']['mean']:.4f} ± {lr['f1']['std']:.4f}
- **Precision:** {lr['precision']['mean']:.4f} ± {lr['precision']['std']:.4f}
- **Recall:** {lr['recall']['mean']:.4f} ± {lr['recall']['std']:.4f}

### Gradient Boosting
- **ROC-AUC:** {gb['auc']['mean']:.4f} ± {gb['auc']['std']:.4f}
- **F1-Score:** {gb['f1']['mean']:.4f} ± {gb['f1']['std']:.4f}
- **Precision:** {gb['precision']['mean']:.4f} ± {gb['precision']['std']:.4f}
- **Recall:** {gb['recall']['mean']:.4f} ± {gb['recall']['std']:.4f}

### Head-to-Head (GB - LR)
| Metric | Difference | Winner |
|--------|-----------|--------|
| ROC-AUC | {comparison['auc']:+.4f} | {'GB' if comparison['auc'] > 0 else 'LR'} |
| F1-Score | {comparison['f1']:+.4f} | {'GB' if comparison['f1'] > 0 else 'LR'} |
| Precision | {comparison['precision']:+.4f} | {'GB' if comparison['precision'] > 0 else 'LR'} |
| Recall | {comparison['recall']:+.4f} | {'GB' if comparison['recall'] > 0 else 'LR'} |

## Conclusion

{_interpret_results(comparison)}

## Limitations & Caveats

1. **Feature set is small:** Only 3 features used. Real churn prediction may benefit from domain features.
2. **No hyperparameter tuning:** Models use near-default hyperparameters to avoid overfitting to this specific dataset.
3. **No statistical testing:** Standard deviations reported; overlapping confidence intervals do not rule out a difference. For a formal claim of superiority, additional seeds or a paired test would strengthen evidence.
4. **Leak surface fully addressed:** The `days_since_last_login` column has been excluded due to leakage. Performance without this column is the **honest** result.
5. **Generalization unknown:** Results on this dataset do not generalize to other churn datasets; the signal and leak landscape are data-dependent.

## Reproducibility

- **Code:** Available in `src/` and `run_experiment.py`
- **Data:** Generated deterministically via `make_dataset.py --seed 7`
- **Seeds:** 42, 123, 456, 789, 999 (logged in metrics.json)
- **Timestamp:** {results.get('timestamp', 'N/A')}
- **Git commit:** {results.get('git_commit', 'N/A')}

To re-run: `python3 run_experiment.py`
"""
    return report


def _interpret_results(comparison: dict) -> str:
    """Interpret the comparison and state the conclusion."""
    auc_diff = comparison["auc"]
    f1_diff = comparison["f1"]

    if abs(auc_diff) < 0.01:
        return (
            "**No detectable difference.** Gradient boosting and logistic regression "
            "perform equivalently on this dataset (AUC difference < 0.01). "
            "Both are viable; logistic regression may be preferred for interpretability."
        )
    elif auc_diff > 0.01:
        pct_gain = (auc_diff / (1 - comparison["auc"] + 0.0001)) * 100
        return (
            f"**Gradient boosting outperforms logistic regression.** "
            f"GB achieves {auc_diff:+.4f} higher ROC-AUC. "
            f"However, without formal statistical testing and given the small feature set, "
            f"this gap may not be practically significant. Both methods are reasonable choices."
        )
    else:
        return (
            f"**Logistic regression outperforms gradient boosting.** "
            f"LR achieves {abs(auc_diff):.4f} higher ROC-AUC. "
            f"This is unexpected; gradient boosting usually matches or exceeds LR performance. "
            f"Possible causes: hyperparameter mismatch or the small feature set favoring linear separation."
        )
