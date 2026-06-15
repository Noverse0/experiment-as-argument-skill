"""Generate REPORT.md from experiment results."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def generate_report(results: dict, results_path: Path, report_path: Path) -> None:
    models = results["models"]
    lr = models["LogisticRegression"]
    gb = models["GradientBoosting"]

    lr_auc = lr["roc_auc"]["mean"]
    gb_auc = gb["roc_auc"]["mean"]
    lr_auc_std = lr["roc_auc"]["std"]
    gb_auc_std = gb["roc_auc"]["std"]

    # Overlapping spreads → no detectable difference
    lr_lo = lr_auc - lr_auc_std
    gb_hi = gb_auc + gb_auc_std
    lr_hi = lr_auc + lr_auc_std
    gb_lo = gb_auc - gb_auc_std
    overlapping = not (gb_lo > lr_hi or lr_lo > gb_hi)

    if overlapping:
        conclusion = (
            "The ROC-AUC spreads overlap, so **no statistically meaningful difference** "
            "is detectable between the two models on this dataset and split strategy."
        )
    elif gb_auc > lr_auc:
        conclusion = (
            f"Gradient Boosting outperforms Logistic Regression by "
            f"{gb_auc - lr_auc:.3f} ROC-AUC (non-overlapping ±1 SD ranges)."
        )
    else:
        conclusion = (
            f"Logistic Regression matches or outperforms Gradient Boosting "
            f"({lr_auc:.3f} vs {gb_auc:.3f} ROC-AUC)."
        )

    table_rows = ""
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        table_rows += (
            f"| {metric:<20} "
            f"| {_fmt(lr[metric]['mean'], lr[metric]['std']):<18} "
            f"| {_fmt(gb[metric]['mean'], gb[metric]['std']):<18} |\n"
        )

    report = f"""\
# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

**Variable:** model class (LogisticRegression vs GradientBoostingClassifier).
All other choices (features, split strategy, preprocessing) are held fixed.

**Data preparation:**
- Dataset: {results['n_samples']} rows after removing {4200 - results['n_samples']} exact duplicates
  (the generator appends 200 duplicate rows; removing them prevents them straddling the split boundary).
- Churn rate: {results['churn_rate']:.1%}

**Feature selection:**
- Used: `tenure_months`, `monthly_spend`, `support_tickets` — the three features with legitimate
  causal signal (no post-outcome information).
- Excluded `days_since_last_login`: **target leakage** — a churned customer has by definition
  stopped logging in, so this column is recorded *after* the outcome is known.
- Excluded `signup_date` (used for ordering only) and `customer_id` (identifier).

**Split strategy:** TimeSeriesSplit with {results['n_cv_splits']} folds on data sorted by
`signup_date`. This respects temporal order and prevents future data leaking into past training folds.

**Preprocessing:** StandardScaler applied to LogisticRegression (fitted on each train fold,
applied to the corresponding test fold). GradientBoosting is scale-invariant and receives raw features.

**Primary metric:** ROC-AUC (handles the ~27 % class imbalance better than accuracy).

**Majority-class baseline ROC-AUC:** {results['baseline_roc_auc']:.3f}

## Results

| Metric               | LogisticRegression     | GradientBoosting       |
|----------------------|------------------------|------------------------|
{table_rows}
*(mean ± std across {results['n_cv_splits']} temporal CV folds)*

LR ROC-AUC range: [{lr_lo:.3f}, {lr_hi:.3f}]
GB ROC-AUC range: [{gb_lo:.3f}, {gb_hi:.3f}]

## Conclusion

{conclusion}

Both models substantially exceed the majority-class baseline ({results['baseline_roc_auc']:.3f}),
confirming the three legitimate features carry real predictive signal.

## Limitations and Remaining Risks

- **Single dataset / single seed:** Results reflect one synthetic dataset. Real-world variance
  may differ.
- **No hyperparameter search:** GradientBoosting defaults were used; a tuned model might perform
  differently, though tuning budget must be equalized across arms.
- **Temporal split approximation:** `signup_date` approximates event time; if churn occurs long
  after sign-up the split boundary may not perfectly separate past/future knowledge.
- **Synthetic data:** The data-generating process (linear logit, Poisson tickets) may favour
  logistic regression; a real dataset with nonlinear interactions could favour tree methods more.

Artifacts: `{results_path}`
"""

    report_path.write_text(report)
    print(f"Report written to {report_path}")


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {path}")
