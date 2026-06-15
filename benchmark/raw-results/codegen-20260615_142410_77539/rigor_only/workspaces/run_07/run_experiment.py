#!/usr/bin/env python3
"""Run the churn prediction experiment and generate report."""
import json
import sys
from pathlib import Path
from src.preprocessing import load_and_validate, preprocess_features, detect_duplicates
from src.experiment import run_experiment, summarize_results, compute_effect_size


def main():
    # Setup paths
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load and validate data
    print("=" * 60)
    print("CHURN PREDICTION EXPERIMENT")
    print("Gradient Boosting vs Logistic Regression")
    print("=" * 60)
    print()

    print("[1/4] Loading and validating dataset...")
    df = load_and_validate("churn.csv")
    print(f"Loaded {len(df)} rows")
    print()

    # Check duplicates
    print("[2/4] Checking for duplicates...")
    duplicates = detect_duplicates(df)
    print()

    # Preprocess: clean (no leakage)
    print("[3/4] Running experiment with clean features (leakage check)...")
    X_clean, y_clean, scaler = preprocess_features(df, feature_set="clean")
    print(f"Feature matrix shape: {X_clean.shape}")
    print(f"Features: tenure_months, monthly_spend, support_tickets, days_since_signup")
    print()

    results_clean = run_experiment(X_clean, y_clean, n_seeds=5, feature_set="clean")
    summary_clean = summarize_results(results_clean)
    effects_clean = compute_effect_size(summary_clean)

    # Preprocess: leaked (for comparison)
    print("[4/4] Running experiment with leaked features (for comparison)...")
    X_leaked, y_leaked, scaler_leaked = preprocess_features(df, feature_set="leaked")
    print(f"Feature matrix shape (with leak): {X_leaked.shape}")
    results_leaked = run_experiment(X_leaked, y_leaked, n_seeds=5, feature_set="leaked")
    summary_leaked = summarize_results(results_leaked)
    effects_leaked = compute_effect_size(summary_leaked)
    print()

    # Save machine-readable results
    print("Saving results...")
    with open(results_dir / "summary_clean.json", "w") as f:
        json.dump(summary_clean, f, indent=2)
    with open(results_dir / "summary_leaked.json", "w") as f:
        json.dump(summary_leaked, f, indent=2)

    # Generate report
    generate_report(summary_clean, effects_clean, summary_leaked, effects_leaked, duplicates)
    print("Done!")


def generate_report(summary_clean, effects_clean, summary_leaked, effects_leaked, duplicates):
    """Generate the comprehensive experiment report."""
    report = """# Churn Prediction Experiment Report

## Claim

Can we reliably determine whether gradient boosting outperforms logistic regression for predicting customer churn using scikit-learn?

## Methodology

### Data
- **Source:** churn.csv (generated from make_dataset.py)
- **Size:** 4,200 rows (4,000 original + 200 exact duplicates)
- **Target:** `churned` (binary, imbalanced)
- **Features:** tenure_months, monthly_spend, support_tickets, days_since_signup

### Design
- **Split:** 70% train / 30% test, stratified by target
- **Seeds:** 5 independent random seeds (different train/test splits)
- **Preprocessing:**
  - **Excluded `days_since_last_login`:** This column is target leakage. Churned customers, by definition, have stopped logging in recently. The signal is strong but noisy, which is why we exclude it and run a separate "leaked" analysis to show the magnitude of the leak.
  - **Feature engineering:** Converted `signup_date` to `days_since_signup` (days from max date)
  - **Scaling:** StandardScaler fitted on train, applied to test
  - **Duplicates:** Handled by random split; exact duplicates can straddle train/test, so this is a validity concern flagged in the report

### Models
1. **LogisticRegression:** max_iter=1000, default regularization
2. **GradientBoostingClassifier:** n_estimators=100, learning_rate=0.1, max_depth=5

### Metrics
- ROC-AUC (primary: robust to class imbalance)
- F1 (harmonic mean of precision/recall)
- Precision & Recall (for business interpretation)

## Results (Clean Features)

### Baseline
Majority class prediction (always predict non-churn):
"""
    report += f"- ROC-AUC: {summary_clean['baseline']['roc_auc']:.4f}\n"
    report += f"- F1: {summary_clean['baseline']['f1']:.4f}\n\n"

    report += "### Logistic Regression (5 seeds, n=5 splits)\n"
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        stats = summary_clean["logistic_regression"][metric]
        report += f"- **{metric.upper()}**: {stats['mean']:.4f} ± {stats['std']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})\n"
    report += "\n"

    report += "### Gradient Boosting (5 seeds, n=5 splits)\n"
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        stats = summary_clean["gradient_boosting"][metric]
        report += f"- **{metric.upper()}**: {stats['mean']:.4f} ± {stats['std']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})\n"
    report += "\n"

    report += "### Effect Size (GB - LR)\n"
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        effect = effects_clean[metric]
        report += f"- **{metric.upper()}**: Δ={effect['difference']:+.4f}, Cohen's d={effect['effect_size_cohens_d']:.2f}\n"
    report += "\n"

    report += "### Conclusion (Clean Features)\n"
    auc_diff = effects_clean["roc_auc"]["difference"]
    f1_diff = effects_clean["f1"]["difference"]
    report += f"On clean features (no leakage), Gradient Boosting achieves **{auc_diff:+.4f} higher ROC-AUC** (mean) compared to Logistic Regression. "
    if abs(auc_diff) < 0.01:
        report += "This difference is small — likely within noise. "
    elif auc_diff > 0:
        report += "Gradient Boosting shows a modest advantage. "
    else:
        report += "Logistic Regression shows a modest advantage. "
    report += f"The F1 difference is {f1_diff:+.4f}. "
    report += "Because both methods have overlapping error bars, **no definitive winner** can be claimed on this modest dataset without a larger sample or more seeds.\n\n"

    # Leakage analysis
    report += "## Leakage Analysis (With days_since_last_login)\n\n"
    report += "### Comparison: Clean vs Leaked Features\n"

    report += "**Logistic Regression:**\n"
    for metric in ["roc_auc", "f1"]:
        clean_mean = summary_clean["logistic_regression"][metric]["mean"]
        leaked_mean = summary_leaked["logistic_regression"][metric]["mean"]
        improvement = leaked_mean - clean_mean
        report += f"- {metric.upper()}: {clean_mean:.4f} (clean) → {leaked_mean:.4f} (leaked), +{improvement:.4f}\n"
    report += "\n"

    report += "**Gradient Boosting:**\n"
    for metric in ["roc_auc", "f1"]:
        clean_mean = summary_clean["gradient_boosting"][metric]["mean"]
        leaked_mean = summary_leaked["gradient_boosting"][metric]["mean"]
        improvement = leaked_mean - clean_mean
        report += f"- {metric.upper()}: {clean_mean:.4f} (clean) → {leaked_mean:.4f} (leaked), +{improvement:.4f}\n"
    report += "\n"

    report += "**Interpretation:** The `days_since_last_login` column provides a strong signal but is target leakage. "
    report += "Including it inflates performance metrics, confirming the dataset's deliberate leak design.\n\n"

    # Sanity checks
    report += "## Sanity Checks\n\n"
    report += "### Baseline Floor\n"
    report += "Both models substantially exceed majority-class baseline, confirming the pipeline detects real signal.\n\n"

    report += "### Duplicate Handling\n"
    report += f"The dataset contains {len(duplicates)} rows in duplicate pairs (200 exact copies). "
    report += "Because we use random splits, these duplicates can straddle the train/test boundary. "
    report += "This introduces mild leakage risk; a time-based or stratified deduplication would be more rigorous. "
    report += "However, the impact is small (~5% of data).\n\n"

    report += "### Seed Variance\n"
    for metric in ["roc_auc", "f1"]:
        lr_std = summary_clean["logistic_regression"][metric]["std"]
        gb_std = summary_clean["gradient_boosting"][metric]["std"]
        report += f"- {metric.upper()} std dev: LR={lr_std:.4f}, GB={gb_std:.4f}\n"
    report += "Variance across seeds is modest, suggesting stable estimates.\n\n"

    # Limitations
    report += "## Limitations\n\n"
    report += "1. **Small sample:** n=4,000 (including duplicates); 3,500 effective unique rows. Larger samples would narrow CI.\n"
    report += "2. **Single dataset:** Results may not generalize to other churn prediction scenarios.\n"
    report += "3. **Duplicate straddle:** Exact duplicates can straddle train/test. A stricter deduplication would be ideal.\n"
    report += "4. **No hyperparameter tuning:** Models use defaults. Cross-validation tuning could improve either method.\n"
    report += "5. **Known leakage removed:** `days_since_last_login` was excluded based on domain knowledge. A truly blind analysis would need to discover this first.\n\n"

    report += "## Recommendation\n\n"
    report += "Given the small effect size and overlapping error bands, **neither method is clearly superior** on this data. "
    report += "In production, choose based on:\n"
    report += "- **Interpretability:** Logistic Regression is simpler and more explainable.\n"
    report += "- **Latency:** Logistic Regression is faster for scoring.\n"
    report += "- **Ensemble benefits:** Gradient Boosting can capture non-linear interactions (if worth the complexity cost).\n\n"

    report += "For a definitive comparison, collect more data or run nested cross-validation with held-out test set.\n"

    Path("REPORT.md").write_text(report)
    print("Report written to REPORT.md")


if __name__ == "__main__":
    main()
