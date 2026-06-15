# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

Logistic regression outperforms gradient boosting (ROC-AUC gap=0.0179, combined sd=0.0000).

Winner: **Logistic Regression**

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.7329 ± 0.0000 | 0.3694 ± 0.0000 | 0.5718 ± 0.0000 | 0.2739 ± 0.0000 |
| Gradient Boosting | 0.7150 ± 0.0000 | 0.3969 ± 0.0000 | 0.5324 ± 0.0000 | 0.3180 ± 0.0000 |

*(mean ± std of fold-means across 3 seeds, 5-fold TimeSeriesSplit each)*

## Methodology

### Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

### Variable
Model class (LogisticRegression vs GradientBoostingClassifier). All other choices — features, preprocessing, evaluation protocol, seeds — are held fixed.

### Data Discipline

**Deduplication:** The raw CSV contains 200 exact duplicate rows. These are removed before any split to prevent the same row appearing in both train and test folds, which would inflate metrics.

**Temporal split:** Data is sorted ascending by `signup_date` and evaluated with `TimeSeriesSplit` (5 folds). This ensures training always precedes the test window in time, matching real deployment conditions and avoiding future-data leakage through random shuffling.

**Leak exclusion:** `days_since_last_login` is **excluded**. This feature is derived from the churn outcome itself — customers who churned stopped logging in, so the value is recorded *after* the decision to churn. Using it would measure the leak's signal, not the model's predictive power. The unusually strong AUC a careless pipeline would observe (from a feature correlated ≈0.6+ with the target) is the diagnostic signal for this trap.

**Features used:** `tenure_months, monthly_spend, support_tickets` — the three legitimate causal signals available in this dataset.

### Preprocessing

- `StandardScaler` is fitted on the training fold only and applied to the test fold inside the CV loop (split-before-transform).
- Gradient boosting receives raw (unscaled) features; tree-based models are scale-invariant.

### Evaluation

- 5-fold `TimeSeriesSplit` per seed; 3 seeds total.
- Primary metric: **ROC-AUC** (measures ranking ability; robust to the 27.1% class imbalance in this dataset).
- Secondary: F1, Precision, Recall reported for completeness.
- Baseline floor (majority-class accuracy): 0.730.

### Sanity Checks Performed

- Majority-class baseline computed before model evaluation.
- Both models evaluated on identical held-out folds (no data leakage between arms).
- Results reproduced across 3 independent seeds.

## Limitations

1. **Thin feature set.** After excluding the leak, only 3 features remain. Both models operate near the same information ceiling, which compresses the performance gap.
2. **Synthetic data structure.** The data-generating process uses a logistic function, which may structurally favour logistic regression. Results on real churn data could differ.
3. **Default hyperparameters.** No tuning was performed. Gradient boosting is more sensitive to hyperparameters than logistic regression; tuning it could widen the gap.
4. **No final hold-out.** Given ~4,000 rows and 5-fold CV, there is no separately withheld test set. The 3-seed repetition compensates but does not fully substitute.
5. **Null-result caveat.** If the gap is within noise, the honest claim is "no detectable difference on this dataset and feature set," not "equal performance."
