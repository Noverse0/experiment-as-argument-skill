# Customer Churn Prediction: LogisticRegression vs GradientBoosting

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Design
- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)
- **Fixed:** Data split (80/20 train/test, stratified), preprocessing (StandardScaler), features (tenure_months, monthly_spend, support_tickets)
- **Repetitions:** 5 random seeds
- **Metric:** ROC-AUC (robust to class imbalance), plus accuracy and PR-AUC for context
- **Test set:** Touched once at the end. All preprocessing fit on train only.

## Data Hygiene
- **Duplicates detected and removed:** 200
- **Features dropped (justification):**
  - `account_status`: Leaked from target ("closed" iff churned==1)
  - `signup_date`: Temporal column; random split ignores time ordering
  - `customer_id`: Identifier, not predictive
- **Features kept:**
  - `tenure_months`: Months as customer (predictive)
  - `monthly_spend`: Monthly spending (predictive)
  - `support_tickets`: Support tickets (predictive)

## Results

### ROC-AUC (primary metric)
| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| LogisticRegression | 0.7389 | 0.0061 | 0.7311 | 0.7498 |
| GradientBoosting | 0.7330 | 0.0035 | 0.7291 | 0.7391 |

### Effect Size
- **Mean difference (GB - LR):** -0.0059
- **Cohen's d:** -1.1907
- **Confidence intervals overlap:** True

### Accuracy and PR-AUC (for reference)
| Model | Accuracy (mean ± std) | PR-AUC (mean ± std) |
|-------|---|---|
| LogisticRegression | 0.7540 ± 0.0065 | 0.5059 ± 0.0153 |
| GradientBoosting | 0.7517 ± 0.0021 | 0.4974 ± 0.0094 |

## Conclusion
**No detectable difference** between models (Δ AUC = -0.0059 with overlapping confidence intervals).

## Limitations & Next Steps
- **Sample size:** 3,360 examples per seed (after dedup). Larger datasets could refine effect size estimates.
- **Hyperparameter tuning:** Models use defaults. A tuning budget (e.g., on a validation set) could improve both.
- **Temporal aspect:** `signup_date` was dropped. If the task is forward-looking, a time-based split should be used instead.
- **Feature engineering:** Only raw features used. Domain-driven features (e.g., spend-per-tenure ratio) might improve both models.
- **Statistical test:** For a formal p-value, a permutation test or bootstrapped CI could be used (deferred).

## Artifacts
- `results/results.json`: Machine-readable metrics (mean, std, min, max per model per seed)
- `results/metrics_by_seed.csv`: Raw metrics for each seed
- `REPORT.md`: This human-readable summary
