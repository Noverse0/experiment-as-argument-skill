"""Generate REPORT.md from experiment results."""
from __future__ import annotations


_CONCLUSION_TEXT = {
    "gb_better": (
        "**Gradient Boosting outperforms Logistic Regression** on this dataset. "
        "The AUC gap exceeds the combined run-to-run spread, indicating a detectable difference."
    ),
    "lr_better": (
        "**Logistic Regression outperforms Gradient Boosting** on this dataset. "
        "The AUC gap exceeds the combined run-to-run spread, indicating a detectable difference."
    ),
    "no_detectable_difference": (
        "**No detectable difference** between the two models. "
        "The AUC gap is within the combined run-to-run spread; the honest claim is that "
        "neither model reliably outperforms the other on this dataset and split."
    ),
}


def write_report(results: dict, path: str = "REPORT.md") -> None:
    meta = results.get("data_meta", {})
    sanity = results.get("sanity", {})
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    cmp = results["comparison"]

    shuffle_status = (
        "PASS (AUC ≈ 0.5)" if sanity.get("shuffle_test_passes", False) else "FAIL"
    )

    conclusion_text = _CONCLUSION_TEXT.get(cmp["conclusion"], cmp["conclusion"])

    lines = [
        "# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression",
        "",
        "## Claim",
        "",
        "Does gradient boosting outperform logistic regression for predicting customer "
        "churn on the provided dataset, when evaluated on a held-out temporal split "
        "using only causally valid features?",
        "",
        "## Methodology",
        "",
        "### Variable",
        "Model family (LogisticRegression vs GradientBoostingClassifier). All other "
        "choices — features, split, preprocessing, hyperparameters — are held fixed.",
        "",
        "### Data Preparation",
        f"- Original rows: {meta.get('original_size', 'N/A')}",
        f"- After deduplication: {meta.get('deduped_size', 'N/A')} "
        f"({meta.get('duplicates_removed', 0)} exact duplicates removed before splitting "
        "to prevent straddling)",
        "",
        "### Feature Exclusions",
        "| Column | Decision | Reason |",
        "|--------|----------|--------|",
        "| `customer_id` | Dropped | Identifier, not a predictor |",
        "| `days_since_last_login` | **Dropped (target leak)** | Recorded at/after the churn event: a churned customer has stopped logging in, so a high value directly encodes the outcome. Including it inflates performance in a way that does not transfer to production. |",
        "| `signup_date` | Converted to `signup_days` | Numeric days since 2023-01-01; legitimate feature fixed at signup time |",
        "",
        "### Features Used",
        "- `tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`",
        "",
        "### Split Strategy",
        f"- **Time-based**: sort by `signup_date`, first {int(meta.get('train_frac', 0.75)*100)}% → train, "
        f"last {100 - int(meta.get('train_frac', 0.75)*100)}% → test",
        f"- Train: {meta.get('train_size', 'N/A')} rows (churn rate: {meta.get('train_churn_rate', 0):.3f})",
        f"- Test: {meta.get('test_size', 'N/A')} rows (churn rate: {meta.get('test_churn_rate', 0):.3f})",
        "- Rationale: random splits on temporal data allow future information to leak into "
        "the training fold; time-based splits simulate the production deployment setting.",
        "",
        "### Preprocessing",
        "- `StandardScaler` fitted on the training fold only, applied to test.",
        "",
        "### Evaluation",
        "- Metrics: AUC-ROC (primary, imbalance-robust) and F1-score (threshold-dependent)",
        f"- Runs: {lr['n_seeds']} seeds per model (seeds: {lr['seeds']})",
        "- LogisticRegression is deterministic given fixed data → std ≈ 0 is expected",
        "",
        "## Sanity Checks",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Baseline AUC floor (majority-class predictor) | {sanity.get('baseline_auc', 'N/A'):.4f} |",
        f"| Label-shuffle AUC (LR on permuted labels) | {sanity.get('label_shuffle_auc', 'N/A'):.4f} — {shuffle_status} |",
        "",
        "- Both models must exceed the baseline AUC of ~0.5.",
        "- The label-shuffle AUC near 0.5 confirms no spurious feature–label correlation "
        "in the clean feature set.",
        "",
        "## Results",
        "",
        "| Model | AUC mean ± std | F1 mean ± std | n seeds |",
        "|-------|---------------|--------------|---------|",
        f"| Logistic Regression | {lr['auc_mean']:.4f} ± {lr['auc_std']:.4f} | {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f} | {lr['n_seeds']} |",
        f"| Gradient Boosting | {gb['auc_mean']:.4f} ± {gb['auc_std']:.4f} | {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f} | {gb['n_seeds']} |",
        "",
        f"AUC gap (GB − LR): **{cmp['auc_gap']:+.4f}** (combined spread: {cmp['combined_spread']:.4f})",
        "",
        "## Conclusion",
        "",
        conclusion_text,
        "",
        "## Limitations",
        "",
        "- **Single dataset, single split**: conclusions are dataset-specific. The underlying "
        "data-generating process is logistic, which structurally favors logistic regression.",
        "- **No hyperparameter tuning**: both models use default/fixed hyperparameters. "
        "Tuning GB (n_estimators, depth, learning rate) could change the result.",
        "- **LR variance is zero by design**: with fixed data and solver, LR is deterministic. "
        "Running 5 seeds confirms reproducibility but does not add statistical power.",
        "- **Time-based split may introduce distribution shift**: customers who signed up later "
        "may have different characteristics than earlier cohorts, biasing test-set estimates.",
        "- **Moderate dataset size**: ~3,800 rows after deduplication limits the power to "
        "detect small differences.",
        "",
        "## Experiment Config",
        "",
        "```",
        "Seeds: [0, 1, 2, 3, 4]",
        "LR: C=1.0, max_iter=1000",
        "GB: n_estimators=100, max_depth=3, learning_rate=0.1",
        "Split: 75/25 time-based on signup_date",
        "Scaler: StandardScaler (train-only fit)",
        "```",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
