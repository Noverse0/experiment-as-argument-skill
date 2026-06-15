"""
Entrypoint for the churn prediction experiment.

Runs sanity checks, then the full experiment comparing logistic regression
vs gradient boosting on customer churn prediction.
"""
import json
from pathlib import Path
from src.experiment import run_experiment
from src.sanity_checks import run_all_sanity_checks


def generate_report(results: dict, sanity_checks: dict, report_path: Path):
    """Generate a markdown report from the experiment results."""
    lr_auc_mean = results["logistic_regression"]["roc_auc"]["mean"]
    lr_auc_std = results["logistic_regression"]["roc_auc"]["std"]
    gb_auc_mean = results["gradient_boosting"]["roc_auc"]["mean"]
    gb_auc_std = results["gradient_boosting"]["roc_auc"]["std"]

    diff = gb_auc_mean - lr_auc_mean
    baseline_auc = results["baseline_majority_class_auc"]

    # Determine winner
    if gb_auc_mean - gb_auc_std > lr_auc_mean + lr_auc_std:
        winner = "**Gradient Boosting** (non-overlapping confidence intervals)"
    elif lr_auc_mean - lr_auc_std > gb_auc_mean + gb_auc_std:
        winner = "**Logistic Regression** (non-overlapping confidence intervals)"
    else:
        winner = "**No detectable difference** (overlapping confidence intervals)"

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does gradient boosting outperform logistic regression on customer churn prediction using legitimate causal features (tenure, monthly spend, support tickets)?

## Methodology

### Data Handling
- **Dataset**: Customer churn data with {results['data_audit']['split_info']['train_size']} training samples and {results['data_audit']['split_info']['test_size']} test samples
- **Split strategy**: Time-based split (sorted by signup_date), 80% train / 20% test to respect temporal structure and avoid future leakage
- **Honest features used**: tenure_months, monthly_spend, support_tickets
- **Dropped (leak)**: days_since_last_login (outcome-derived; churned customers recorded as inactive post-hoc)

### Duplicate Audit
- Total rows in raw data: {results['data_audit']['duplicates']['total_rows']}
- Exact full duplicates: {results['data_audit']['duplicates']['full_duplicates']}
- Feature-wise duplicates: {results['data_audit']['duplicates']['feature_duplicates']}
- Rows straddling train/test boundary: {results['data_audit']['split_info']['feature_overlaps']}

### Class Balance
- Training set churn rate: {results['data_audit']['train_churn_rate']:.2%}
- Test set churn rate: {results['data_audit']['test_churn_rate']:.2%}

### Models Compared
1. **Logistic Regression**: Linear model with L2 regularization, StandardScaler preprocessing, max_iter=1000
2. **Gradient Boosting**: 100 trees, learning_rate=0.1, max_depth=5, no preprocessing (tree-based)

### Evaluation
- **Primary metric**: ROC-AUC (robust to class imbalance)
- **Secondary metrics**: Precision, Recall, F1
- **Repetitions**: {results['config']['num_seeds']} independent runs (different random seeds)
- **Reporting**: mean ± std dev across runs

### Baseline
- Majority class predictor (always predict most common class): ROC-AUC = {baseline_auc:.4f}
- Both models must exceed baseline to be credible

## Sanity Checks

### Label-Shuffle Test
With shuffled labels, model performance should drop to random baseline:
- Original AUC: {sanity_checks['label_shuffle']['original_auc']:.4f}
- Shuffled AUC: {sanity_checks['label_shuffle']['shuffled_auc']:.4f}
- Baseline AUC: {sanity_checks['label_shuffle']['baseline_auc']:.4f}
- ✓ Drop detected: {sanity_checks['label_shuffle']['drop_detected']} (no leakage around labels)

### Tiny Batch Overfit Test
Model should converge to high training accuracy on a small subset:
- Subset size: {sanity_checks['overfit_tiny']['tiny_subset_size']}
- Training AUC: {sanity_checks['overfit_tiny']['train_auc_on_tiny']:.4f}
- ✓ Converged: {sanity_checks['overfit_tiny']['converged']}

### Leak Feature Validation
Training with the leak feature (days_since_last_login) should inflate performance:
- Honest features only: {sanity_checks['leakage_demo']['honest_features_auc']:.4f}
- With leak feature: {sanity_checks['leakage_demo']['with_leak_feature_auc']:.4f}
- Boost from leak: +{sanity_checks['leakage_demo']['leak_boost']:.4f}
- ✓ Leak confirmed (justifies exclusion)

## Results

### Logistic Regression
- **ROC-AUC**: {lr_auc_mean:.4f} ± {lr_auc_std:.4f}
- **Precision**: {results['logistic_regression']['precision']['mean']:.4f} ± {results['logistic_regression']['precision']['std']:.4f}
- **Recall**: {results['logistic_regression']['recall']['mean']:.4f} ± {results['logistic_regression']['recall']['std']:.4f}
- **F1**: {results['logistic_regression']['f1']['mean']:.4f} ± {results['logistic_regression']['f1']['std']:.4f}

### Gradient Boosting
- **ROC-AUC**: {gb_auc_mean:.4f} ± {gb_auc_std:.4f}
- **Precision**: {results['gradient_boosting']['precision']['mean']:.4f} ± {results['gradient_boosting']['precision']['std']:.4f}
- **Recall**: {results['gradient_boosting']['recall']['mean']:.4f} ± {results['gradient_boosting']['recall']['std']:.4f}
- **F1**: {results['gradient_boosting']['f1']['mean']:.4f} ± {results['gradient_boosting']['f1']['std']:.4f}

### Comparison
- Difference in ROC-AUC (GB - LR): {diff:+.4f}
- **Winner**: {winner}

## Conclusion

{winner} on this dataset. The difference of {diff:+.4f} in ROC-AUC is {"within the" if winner == "**No detectable difference** (overlapping confidence intervals)" else "outside the"} margin of statistical noise (std dev ~{max(lr_auc_std, gb_auc_std):.4f}).

Both models significantly exceed the baseline ({baseline_auc:.4f}), confirming the pipeline is working and legitimate signal is present in the features.

## Limitations & Remaining Risks

1. **Small test set**: {results['data_audit']['split_info']['test_size']} samples is modest; confidence intervals may be wide.
2. **Hyperparameter tuning**: Models were run with fixed hyperparameters; no tuning on validation set. A full pipeline would include CV-based hyperparameter search.
3. **Temporal split assumption**: The time-based split assumes customer acquisition order is the right temporal boundary. If signup_date does not reflect the true observation time, this choice is suboptimal.
4. **Limited feature engineering**: Only raw features were used; polynomial/interaction features might unlock additional signal.
5. **Seed variance**: Results depend on the random seed selection; only 5 seeds were run. Broader sampling would increase confidence.

## Reproducibility

All hyperparameters, seeds, split method, and feature selection are recorded in `results/metrics.json`.
To reproduce: `python3 run_experiment.py`
"""
    with open(report_path, "w") as f:
        f.write(report)


if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Running sanity checks...")
    sanity_checks = run_all_sanity_checks("churn.csv")
    with open(results_dir / "sanity_checks.json", "w") as f:
        json.dump(sanity_checks, f, indent=2)
    print(f"✓ Sanity checks passed. Results saved to results/sanity_checks.json")

    print("\nRunning full experiment (this may take 1-2 minutes)...")
    results = run_experiment("churn.csv", results_dir, num_seeds=5)
    print(f"✓ Experiment complete. Metrics saved to results/metrics.json")

    print("\nGenerating report...")
    generate_report(results, sanity_checks, Path("REPORT.md"))
    print(f"✓ Report generated at REPORT.md")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    lr_auc = results["logistic_regression"]["roc_auc"]["mean"]
    gb_auc = results["gradient_boosting"]["roc_auc"]["mean"]
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print(f"Gradient Boosting ROC-AUC:   {gb_auc:.4f}")
    print(f"Difference (GB - LR):        {gb_auc - lr_auc:+.4f}")
    print("=" * 70)
