# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` for predicting
customer churn on this dataset?

## Methodology

### Variable
Model type (LR vs GBT). All other choices — features, preprocessing, split, seeds — are
held fixed across both arms.

### Features Used
`tenure_months, monthly_spend, support_tickets`

### Features Excluded
| Column | Reason |
|--------|--------|
| `customer_id` | Identifier — not a predictor |
| `signup_date` | Used for temporal split ordering; redundant with `tenure_months` as a feature |
| `days_since_last_login` | **Target leak.** This value is recorded *after* churn occurs: a churned customer has already stopped logging in, so the feature encodes the outcome. Including it yields AUC ≈ 0.96 (sanity-check ceiling), confirming it is not a legitimate predictor. |

### Data Quality
- 200 exact duplicate rows (same `customer_id`) were detected and removed before
  any splitting.

### Split Strategy
Temporal split (80 / 20) sorted by `signup_date`:

- Train: 2023-01-01 → 2024-12-21 (3200 rows)
- Test:  2024-12-21 → 2025-06-18 (800 rows)

Random splits on temporal data risk leakage through near-duplicate neighbors across
the boundary; temporal ordering prevents this.

### Preprocessing
- `LogisticRegression`: `StandardScaler` (fit on train only, applied to test)
- `GradientBoostingClassifier`: no scaling needed

### Evaluation
- Primary metric: **ROC-AUC** (robust to the ~27% churn imbalance)
- Also reported: Average Precision (PR-AUC), F1
- Cross-validation: `RepeatedStratifiedKFold` (5 folds × 3 repeats = 15 evaluations) on training data
- Holdout: single evaluation on test set (touched once, at the end)

---

## Results

### Dataset
| Stat | Value |
|------|-------|
| Rows after dedup | 4000 |
| Overall churn rate | 27.1% |
| Train / Test | 3200 / 800 |
| Baseline AUC (majority class) | 0.5000 |
| Leak-included ceiling AUC | 0.9579 |

### Cross-Validation (n=15 folds, on training data)
| Model | ROC-AUC (mean ± sd) | Avg Precision | F1 |
|-------|---------------------|---------------|----|
| LogisticRegression | 0.7372 ± 0.0153 | 0.5124 ± 0.0307 | 0.3573 ± 0.0334 |
| GradientBoosting | 0.7275 ± 0.0187 | 0.5006 ± 0.0299 | 0.3899 ± 0.0323 |

CV gap (GBT − LR): **-0.0097** (combined SD: 0.0242)

### Holdout Test Set
| Model | ROC-AUC | Avg Precision |
|-------|---------|---------------|
| LogisticRegression | 0.7323 | 0.4922 |
| GradientBoosting | 0.7237 | 0.4812 |

Holdout gap (GBT − LR): **-0.0085**

---

## Conclusion

**No detectable difference.** The gap between GradientBoosting and LogisticRegression is -0.0097 AUC, which is within noise (combined SD: 0.0242). With 15 CV evaluations, the honest conclusion is a tie: neither model measurably outperforms the other on the legitimate features.

---

## Limitations and Validity Threats

1. **Small feature set.** Only three legitimate predictors survived the leak audit.
   The causal signal in this dataset is intentionally weak (`tenure_months`,
   `monthly_spend`, `support_tickets`), which limits the performance ceiling for
   both models and reduces statistical power to detect differences.

2. **No hyperparameter tuning.** Default hyperparameters were used for both models.
   Tuning GBT (e.g., learning rate, depth) might narrow or widen the gap, but
   tuning on any split that touches the test set would convert test into validation.

3. **Single dataset.** Results reflect this synthetic dataset's distribution; they
   may not generalize to real-world churn data with richer feature spaces.

4. **Temporal split approximates deployment.** A rolling-origin evaluation would more
   faithfully simulate production use, but was omitted for simplicity.

5. **`days_since_last_login` exclusion is critical.** Any pipeline that includes this
   column should report AUC ≈ 0.96 as a stop signal — not a result.
