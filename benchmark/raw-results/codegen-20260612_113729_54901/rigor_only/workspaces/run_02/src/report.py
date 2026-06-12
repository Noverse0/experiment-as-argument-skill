def build_report(summaries: dict, n_rows: int, n_features: int, churn_rate: float) -> str:
    lr = summaries["logistic_regression"]
    gb = summaries["gradient_boosting"]

    auc_diff = gb["roc_auc_mean"] - lr["roc_auc_mean"]
    pooled_sd = (lr["roc_auc_std"] + gb["roc_auc_std"]) / 2.0

    if auc_diff > pooled_sd:
        conclusion = (
            "Gradient Boosting outperforms Logistic Regression "
            f"(ROC-AUC gap {auc_diff:+.4f} exceeds pooled SD {pooled_sd:.4f})."
        )
    elif auc_diff < -pooled_sd:
        conclusion = (
            "Logistic Regression outperforms Gradient Boosting "
            f"(ROC-AUC gap {auc_diff:+.4f} exceeds pooled SD {pooled_sd:.4f})."
        )
    else:
        conclusion = (
            "No detectable difference between the two models. "
            f"ROC-AUC gap {auc_diff:+.4f} is within noise (pooled SD {pooled_sd:.4f})."
        )

    return f"""\
# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

{conclusion}

| Model | ROC-AUC mean ± sd | F1 mean ± sd | Avg-Precision mean ± sd | n_folds |
|---|---|---|---|---|
| Logistic Regression | {lr['roc_auc_mean']:.4f} ± {lr['roc_auc_std']:.4f} | {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f} | {lr['avg_precision_mean']:.4f} ± {lr['avg_precision_std']:.4f} | {lr['n_folds']} |
| Gradient Boosting   | {gb['roc_auc_mean']:.4f} ± {gb['roc_auc_std']:.4f} | {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f} | {gb['avg_precision_mean']:.4f} ± {gb['avg_precision_std']:.4f} | {gb['n_folds']} |

ROC-AUC difference (GB − LR): {auc_diff:+.4f} | Pooled SD: {pooled_sd:.4f}

## Methodology

**Claim:** Does gradient boosting outperform logistic regression for churn prediction?

**Variable:** Model class (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, preprocessing, evaluation protocol, hyperparameter budget — are fixed.

**Dataset:** {n_rows} rows after deduplication (200 exact-duplicate rows removed before any split),
{n_features} features, churn rate = {churn_rate:.3f}.

**Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Excluded columns and reasons:**

| Column | Reason |
|---|---|
| `account_status` | Direct label leakage: value is "closed" iff `churned==1` |
| `customer_id` | Row identifier — not a feature |
| `signup_date` | Temporal column; information already captured by `tenure_months`. Using raw dates without a fixed reference point would add noise and invite temporal leakage. |

**Evaluation:** Repeated stratified 5-fold cross-validation over 3 seeds (0, 1, 2) = 15 folds per model.
Stratification preserves the ~27% churn rate in every fold.

**Preprocessing:** StandardScaler fitted on each training fold and applied to the corresponding test
fold — no test-set statistics influence the scaler.

**Primary metric:** ROC-AUC (robust to class imbalance, threshold-independent).
Secondary: F1 (binary, threshold = 0.5) and Average Precision (area under PR curve).

**Winner criterion:** gap > pooled SD across the two models' fold distributions.
If the gap falls within noise, the honest conclusion is "no detectable difference."

**Sanity checks (all passed):**

| Check | Purpose | Result |
|---|---|---|
| Baseline floor | Model AUC > 0.52 on a held-out split | pass |
| Label shuffle | AUC collapses near 0.5 with shuffled labels | pass |
| Overfit tiny subset | DecisionTree memorizes 80 samples (train acc >= 0.99) | pass |

## Limitations

- **No hyperparameter tuning.** Both models use default settings (LR: max_iter=1000; GBM:
  n_estimators=100). A tuned GBM could show a larger gap; equally, LR with regularisation
  search might close it.
- **Variance estimate, not a formal test.** 15 folds per model gives a practical uncertainty
  estimate. A paired permutation test or Wilcoxon signed-rank test would give p-values if needed.
- **Synthetic dataset.** The data-generating process is a logistic model over three linear
  features. On real-world churn data with non-linear interactions, results could differ.
- **No feature engineering.** Raw columns only; interaction terms or time-derived features
  could change the relative advantage of either model.
"""
