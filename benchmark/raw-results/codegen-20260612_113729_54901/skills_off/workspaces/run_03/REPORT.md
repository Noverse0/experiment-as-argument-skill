# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

**Result: Logistic regression outperforms gradient boosting.**
LR ROC-AUC 0.733±0.022 vs GBM 0.675±0.028 (gap -0.058).

## Design

**Variable under test:** Model class (LogisticRegression vs GradientBoostingClassifier).
All other factors are held fixed: same features, same preprocessing pipeline, same CV protocol, same random seed (42).

### Data handling

| Step | Detail |
|------|--------|
| Raw rows | 4200 |
| Exact duplicates removed (before split) | 200 |
| Clean rows used | 4000 |
| Churn rate | 27.1% |

**Leakage mitigations applied:**
- `account_status` **dropped** — perfect leak: value is `"closed"` iff `churned==1`.
- `customer_id` dropped — identifier, no predictive signal.
- 200 exact duplicate rows removed *before* any split to prevent train/test contamination.
- `signup_date` encoded as days since 2023-01-01 (fixed reference; no fitting required).

**Features used:** `tenure_months, monthly_spend, support_tickets, days_since_ref`

### Split policy

5-fold **TimeSeriesSplit** on data sorted ascending by `signup_date`.
Training data always precedes test data temporally, respecting the forward-looking nature of churn prediction and avoiding temporal leakage from random splits.

### Metrics

- **ROC-AUC** (primary) — threshold-independent; handles class imbalance; reported as mean ± std across 5 folds.
- **F1** and **Accuracy** (secondary).

### Models

| Model | Configuration |
|-------|---------------|
| LogisticRegression | StandardScaler + LR(max_iter=1000, random_state=42) |
| GradientBoostingClassifier | StandardScaler + GBM(n_estimators=100, max_depth=3, random_state=42) |
| Majority baseline | DummyClassifier(strategy="most_frequent") |

## Sanity Checks

- Duplicate rows removed before splitting — confirmed by data_stats.
- `account_status` excluded — confirmed by feature list above.
- Both trained models exceed ROC-AUC 0.5; majority baseline cannot be computed for ROC-AUC (no probability output).
- Baseline accuracy reported to verify trained models improve over trivial prediction.

## Results

| Model | ROC-AUC mean±sd | F1 mean±sd | Accuracy mean±sd |
|-------|-----------------|------------|------------------|
| logistic_regression | 0.733±0.022 | 0.349±0.056 | 0.750±0.021 |
| gradient_boosting | 0.675±0.028 | 0.400±0.089 | 0.669±0.103 |
| majority_baseline | N/A | 0.000±0.000 | 0.731±0.019 |

Per-fold ROC-AUC values:
- LR: [0.7323925314806774, 0.7376626506024095, 0.6959162303664922, 0.7659099281635662, 0.7320481927710843]
- GB: [0.664681936604429, 0.6352951807228915, 0.6626563791678147, 0.7105878154356234, 0.7013192771084338]

**AUC gap (GB − LR):** -0.058
**Noise threshold (max fold std):** 0.028

## Conclusion

**Logistic regression outperforms gradient boosting.**
LR ROC-AUC 0.733±0.022 vs GBM 0.675±0.028 (gap -0.058).

## Limitations

1. **Single random seed (42).** Variance is estimated across 5 CV folds, but not across multiple seeds. Running 3–5 seeds would sharpen the conclusion.
2. **No hyperparameter tuning.** Default parameters used. GBM may benefit from tuning `n_estimators`, `learning_rate`, or `max_depth`; LR from tuning `C`.
3. **Synthetic data with a logistic DGP.** The true data-generating process is `logit = f(tenure, spend, tickets)` — linear in the log-odds. This structurally favors logistic regression. On real-world churn data (non-linear interactions, missing values, high cardinality categoricals) GBM may outperform more clearly.
4. **Fixed temporal window.** The dataset spans 2023–2025. Generalization to distributions outside this window is not tested.
5. **No feature engineering.** Domain-specific features (e.g., spend-per-month trends, ticket rate) were not constructed. GBM in particular may benefit from richer features.
