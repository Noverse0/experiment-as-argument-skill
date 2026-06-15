#!/usr/bin/env python3
"""Entrypoint: generate dataset, run experiment, write results and report."""
import subprocess
import json
from pathlib import Path
import sys

from src.dataset import load_and_prepare
from src.experiment import run_experiment


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("=== Generating dataset ===")
    subprocess.run(["python3", "make_dataset.py", "--out", "churn.csv"], check=True)

    # Step 2: Load and prepare data
    print("=== Loading and preparing data ===")
    X, y, signup_dates, metadata = load_and_prepare("churn.csv")
    print(f"Dataset shape: {X.shape}")
    print(f"Duplicates removed: {metadata['n_duplicates_removed']}")
    print(f"Class distribution: {metadata['class_distribution']}")

    # Step 3: Run experiment
    print("=== Running experiment (5 seeds) ===")
    results = run_experiment(X, y, signup_dates, n_runs=5)

    # Step 4: Save machine-readable results
    print("=== Saving results ===")
    results_to_save = {
        "metadata": metadata,
        "results": {},
    }
    for model_name, metrics in results.items():
        results_to_save["results"][model_name] = {
            metric: {
                "mean": float(data["mean"]),
                "std": float(data["std"]),
                "n": len(data["values"]),
            }
            for metric, data in metrics.items()
        }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    # Step 5: Generate report
    print("=== Generating report ===")
    report = generate_report(results, metadata)
    with open("REPORT.md", "w") as f:
        f.write(report)

    print("\n✓ Results saved to results/metrics.json")
    print("✓ Report saved to REPORT.md")


def generate_report(results, metadata):
    """Generate markdown report with comparison conclusion."""
    lr_auc = results["LogisticRegression"]["auc"]["mean"]
    gb_auc = results["GradientBoostingClassifier"]["auc"]["mean"]
    auc_diff = gb_auc - lr_auc

    auc_lr_std = results["LogisticRegression"]["auc"]["std"]
    auc_gb_std = results["GradientBoostingClassifier"]["auc"]["std"]

    # Determine if the difference is meaningful (outside error margins)
    is_meaningful = auc_diff > (auc_lr_std + auc_gb_std)

    report = f"""# Churn Prediction Experiment Report

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- Original rows: {metadata['n_rows_original']}
- Duplicates removed: {metadata['n_duplicates_removed']}
- Clean rows: {metadata['n_rows_clean']}
- Class distribution: {metadata['class_distribution']}

### Features (Explicitly Curated)
**Included:** {', '.join(metadata['features'])}
- These are the "honest" causal features in the dataset

**Excluded & Justification:**
- `customer_id`: No predictive signal
- `days_since_last_login`: **Target leak** — churned customers by definition have not logged in recently; this value is recorded at/after the outcome, not before
- `signup_date`: Temporal feature requiring time-based split (included in split logic, not as a feature)

### Split Strategy
- **Time-based split** (80/20): sorted by signup_date to prevent information leakage from temporal ordering
- Prevents future data from bleeding into train set
- Respects the causal direction (predict churn from pre-outcome information)

### Preprocessing
- StandardScaler fitted on train set only, applied to test set
- Ensures no information leakage from scaling statistics

### Models
1. **LogisticRegression** (random_state=seed, max_iter=1000, solver='lbfgs')
2. **GradientBoostingClassifier** (n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)

### Repetition
- 5 independent runs with different seeds (100–104)
- Each run: full pipeline (split, fit, evaluate)
- Report: mean ± std across runs

## Results

### Logistic Regression
| Metric    | Mean   | Std    | Values                                            |
|-----------|--------|--------|---------------------------------------------------|
| AUC       | {results['LogisticRegression']['auc']['mean']:.4f} | {results['LogisticRegression']['auc']['std']:.4f} | {[f'{v:.4f}' for v in results['LogisticRegression']['auc']['values']]} |
| Precision | {results['LogisticRegression']['precision']['mean']:.4f} | {results['LogisticRegression']['precision']['std']:.4f} | {[f'{v:.4f}' for v in results['LogisticRegression']['precision']['values']]} |
| Recall    | {results['LogisticRegression']['recall']['mean']:.4f} | {results['LogisticRegression']['recall']['std']:.4f} | {[f'{v:.4f}' for v in results['LogisticRegression']['recall']['values']]} |
| F1        | {results['LogisticRegression']['f1']['mean']:.4f} | {results['LogisticRegression']['f1']['std']:.4f} | {[f'{v:.4f}' for v in results['LogisticRegression']['f1']['values']]} |
| Accuracy  | {results['LogisticRegression']['accuracy']['mean']:.4f} | {results['LogisticRegression']['accuracy']['std']:.4f} | {[f'{v:.4f}' for v in results['LogisticRegression']['accuracy']['values']]} |

### Gradient Boosting Classifier
| Metric    | Mean   | Std    | Values                                            |
|-----------|--------|--------|---------------------------------------------------|
| AUC       | {results['GradientBoostingClassifier']['auc']['mean']:.4f} | {results['GradientBoostingClassifier']['auc']['std']:.4f} | {[f'{v:.4f}' for v in results['GradientBoostingClassifier']['auc']['values']]} |
| Precision | {results['GradientBoostingClassifier']['precision']['mean']:.4f} | {results['GradientBoostingClassifier']['precision']['std']:.4f} | {[f'{v:.4f}' for v in results['GradientBoostingClassifier']['precision']['values']]} |
| Recall    | {results['GradientBoostingClassifier']['recall']['mean']:.4f} | {results['GradientBoostingClassifier']['recall']['std']:.4f} | {[f'{v:.4f}' for v in results['GradientBoostingClassifier']['recall']['values']]} |
| F1        | {results['GradientBoostingClassifier']['f1']['mean']:.4f} | {results['GradientBoostingClassifier']['f1']['std']:.4f} | {[f'{v:.4f}' for v in results['GradientBoostingClassifier']['f1']['values']]} |
| Accuracy  | {results['GradientBoostingClassifier']['accuracy']['mean']:.4f} | {results['GradientBoostingClassifier']['accuracy']['std']:.4f} | {[f'{v:.4f}' for v in results['GradientBoostingClassifier']['accuracy']['values']]} |

## Conclusion

**AUC Comparison (primary metric):**
- Logistic Regression: {lr_auc:.4f} ± {auc_lr_std:.4f}
- Gradient Boosting: {gb_auc:.4f} ± {auc_gb_std:.4f}
- Difference: {auc_diff:.4f}

**Interpretation:**
"""

    if is_meaningful:
        report += f"The {('Gradient Boosting' if auc_diff > 0 else 'Logistic Regression')} model shows a meaningful performance advantage (difference of {abs(auc_diff):.4f} exceeds the sum of standard errors {auc_lr_std + auc_gb_std:.4f})."
    else:
        report += f"The performance difference is within the noise (difference {auc_diff:.4f} < sum of std errors {auc_lr_std + auc_gb_std:.4f}). **No detectable winner.**"

    report += """

## Validity Checks

✓ Duplicates identified and removed
✓ Target leak explicitly identified and excluded
✓ Time-based split (not random)
✓ Preprocessing fitted on train only
✓ Multiple seeds (5) for stability
✓ Same hyperparameters across seeds
✓ Test set evaluated once (no peeking)

## Limitations

1. **Limited feature engineering:** Only three "honest" features used. More sophisticated feature engineering (e.g., feature interactions, domain-derived features) could improve both models.
2. **Hyperparameter sensitivity:** Models use fixed hyperparameters; tuning could shift the comparison.
3. **Class imbalance:** If present, metrics like accuracy may be misleading (see class distribution above).
4. **Generalization:** Results specific to this dataset; findings may not transfer to other churn datasets.

## Machine-Readable Results
See `results/metrics.json` for full results in JSON format.
"""
    return report


if __name__ == "__main__":
    main()
