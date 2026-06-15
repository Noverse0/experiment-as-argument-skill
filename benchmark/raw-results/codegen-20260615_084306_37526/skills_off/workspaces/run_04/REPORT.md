# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Conclusion
**No detectable difference** between the two models (gap ≤ noise floor).

| Model | AUC-ROC mean ± std | F1 mean ± std | n evals |
|---|---|---|---|
| Majority-class baseline | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 5 |
| Logistic Regression | 0.7329 ± 0.0226 | 0.3694 ± 0.0358 | 25 |
| Gradient Boosting | 0.7150 ± 0.0198 | 0.3969 ± 0.0351 | 25 |

AUC-ROC gap: 0.0179 | Noise floor (max std): 0.0226

## Dataset
- Rows after deduplication: 4000
- Churn rate: 27.1%
- Features used: tenure_months, monthly_spend, support_tickets

## Methodology

### Leak Audit and Feature Selection
Three columns were excluded:

- **`customer_id`**: row identifier, zero predictive signal.
- **`signup_date`**: used to enforce temporal ordering of the split; encoding
  it as a numeric feature would confound cohort membership with model signal,
  so it is excluded from the feature matrix.
- **`days_since_last_login`**: **post-outcome leak.** A churned customer has,
  by definition, stopped logging in. This value is recorded *after* the outcome
  is known, not before. Including it would let the model read the answer from
  the data rather than learn a causal signal. It was dropped to ensure the
  pipeline generalises to the pre-churn decision window where this value is
  not yet observed.

### Deduplication
The dataset contains exact duplicate rows. These were removed before splitting
to prevent any duplicate from appearing in both train and test folds.

### Split Policy
`TimeSeriesSplit(n_splits=5)` over rows sorted by `signup_date`.
Each fold trains on earlier cohorts and tests on later ones — the operationally
realistic scenario where a model trained today predicts churn for future
customers.

### Preprocessing
`StandardScaler` is fitted on the train fold and applied to the test fold
within each cross-validation iteration. Test statistics never influence the
scaler.

### Repetition
5 random seeds × 5 CV folds = **25 evaluations per model**.
Mean ± std is reported. A winner is only claimed when the AUC-ROC gap exceeds
the noise floor (max std of the two models); otherwise the result is declared
"no detectable difference."

### Metrics
- **Primary: AUC-ROC** — threshold-independent, robust to class imbalance.
- **Secondary: F1** — at the default 0.5 threshold; included for completeness.

## Limitations
1. **Synthetic data**: results may not generalise to real churn datasets where
   behavioural sequences, product type, and lifecycle length add complexity.
2. **No hyperparameter tuning**: both models use fixed near-default settings.
   Tuned gradient boosting would likely show a larger advantage if one exists.
3. **F1 is threshold-dependent**: the default 0.5 threshold is arbitrary;
   AUC-ROC is the more reliable comparison metric here.
4. **Short observation window**: all customers in this dataset signed up within
   ~2.5 years; longer temporal drift may alter the comparison.
