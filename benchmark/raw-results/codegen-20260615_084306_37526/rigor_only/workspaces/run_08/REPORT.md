# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

### Variable
Model class — everything else held fixed: same features, same preprocessing (StandardScaler),
same CV folds, same random seeds.

### Features Used
| Feature | Justification |
|---------|--------------|
| `tenure_months` | Customer age; genuine causal signal |
| `monthly_spend` | Revenue proxy; included in DGP signal |
| `support_tickets` | Dissatisfaction proxy; included in DGP signal |

### Excluded Features (Leak Audit)
| Feature | Reason |
|---------|--------|
| `customer_id` | Row identifier — no signal |
| `signup_date` | Temporal column; without a deployment time anchor a random split would be invalid for time-based features |
| `days_since_last_login` | **Target leak** — churned customers stop logging in, so this value is recorded *after* the outcome is known. Including it would inflate AUC artificially without being available at prediction time. |

### Data Cleaning
Removed **200 exact duplicate rows** before any split. Duplicates straddling
train/test in a random split would inflate held-out metrics.

### Evaluation Protocol
- 5-fold stratified cross-validation repeated over 5 seeds: `[0, 1, 2, 3, 4]`
- Total fold evaluations per model: **25**
- Primary metric: **ROC-AUC** (threshold-free, handles class imbalance)
- Secondary metric: **F1** (threshold-sensitive summary)
- Significance: paired t-test on fold-level AUC scores (df = 24)

### Class Balance
Churn rate in cleaned dataset: **27.1%**

## Results

| Model | ROC-AUC mean ± std | F1 mean ± std |
|-------|--------------------|---------------|
| LogisticRegression | 0.7360 ± 0.0134 | 0.3480 ± 0.0277 |
| GradientBoosting | 0.7260 ± 0.0124 | 0.3648 ± 0.0276 |

**AUC gap (GB − LR): -0.0100**
Paired t-test: t = -6.265, p = 0.0000 (significant at α = 0.05)

## Conclusion

**LogisticRegression wins.** GradientBoosting achieves higher AUC by 0.0100 (p = 0.0000 < 0.05). The difference is statistically detectable across 25 fold evaluations.

## Limitations

1. **Temporal validity not tested**: `signup_date` was excluded because a random CV split
   cannot simulate a time-ordered deployment. A production evaluation should train on
   customers who signed up before a cutoff and test on those who signed up after.

2. **No hyperparameter tuning**: GradientBoosting uses default hyperparameters. A tuned
   GB may perform differently relative to tuned LR. Tuning budget should be identical
   for both arms to keep the comparison fair.

3. **Synthetic data**: The DGP uses a simple logistic signal (tenure, spend, tickets)
   that LR is well-specified for. Results may not generalize to real churn datasets
   with non-linear interactions.

4. **Single dataset version**: Results hold for seed=7, n=4200. A different dataset
   realization could shift the ranking.
