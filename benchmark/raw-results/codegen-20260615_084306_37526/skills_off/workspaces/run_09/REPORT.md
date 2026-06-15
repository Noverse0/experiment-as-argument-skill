# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

**Variable:** model class (LogisticRegression vs GradientBoostingClassifier).
All other choices (features, split strategy, preprocessing) are held fixed.

**Data preparation:**
- Dataset: 4000 rows after removing 200 exact duplicates
  (the generator appends 200 duplicate rows; removing them prevents them straddling the split boundary).
- Churn rate: 27.1%

**Feature selection:**
- Used: `tenure_months`, `monthly_spend`, `support_tickets` — the three features with legitimate
  causal signal (no post-outcome information).
- Excluded `days_since_last_login`: **target leakage** — a churned customer has by definition
  stopped logging in, so this column is recorded *after* the outcome is known.
- Excluded `signup_date` (used for ordering only) and `customer_id` (identifier).

**Split strategy:** TimeSeriesSplit with 5 folds on data sorted by
`signup_date`. This respects temporal order and prevents future data leaking into past training folds.

**Preprocessing:** StandardScaler applied to LogisticRegression (fitted on each train fold,
applied to the corresponding test fold). GradientBoosting is scale-invariant and receives raw features.

**Primary metric:** ROC-AUC (handles the ~27 % class imbalance better than accuracy).

**Majority-class baseline ROC-AUC:** 0.500

## Results

| Metric               | LogisticRegression     | GradientBoosting       |
|----------------------|------------------------|------------------------|
| roc_auc              | 0.733 ± 0.023      | 0.715 ± 0.020      |
| f1                   | 0.369 ± 0.036      | 0.397 ± 0.035      |
| precision            | 0.572 ± 0.045      | 0.532 ± 0.044      |
| recall               | 0.274 ± 0.033      | 0.318 ± 0.037      |

*(mean ± std across 5 temporal CV folds)*

LR ROC-AUC range: [0.710, 0.755]
GB ROC-AUC range: [0.695, 0.735]

## Conclusion

The ROC-AUC spreads overlap, so **no statistically meaningful difference** is detectable between the two models on this dataset and split strategy.

Both models substantially exceed the majority-class baseline (0.500),
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

Artifacts: `results/metrics.json`
