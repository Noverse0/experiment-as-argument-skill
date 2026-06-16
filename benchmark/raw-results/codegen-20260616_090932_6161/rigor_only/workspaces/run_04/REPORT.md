# Churn Prediction: LogisticRegression vs GradientBoostingClassifier

**Date:** 2026-06-16 09:51:51

## Conclusion

**Gradient Boosting UNDERPERFORMS Logistic Regression**

- LogisticRegression AUC: `0.7409 ± 0.0000` (n=5)
- GradientBoostingClassifier AUC: `0.7187 ± 0.0004` (n=5)
- Gap: `-0.0222`

## Methodology

### Data
- **Source:** churn.csv
- **Split:** 70% train / 30% test
- **Split method:** time-based (days_since_signup)
- **Duplicates:** Deduplicated before splitting to prevent contamination

### Features
- **Included:** tenure_months, monthly_spend, support_tickets, days_since_signup
- **Removed:** days_since_last_login (target leak); customer_id; signup_date

### Preprocessing
1. Deduplicate exact rows
2. Remove target leakage (days_since_last_login encodes the outcome post-facto)
3. Engineer days_since_signup from temporal column
4. Standard scale features (fit on train, apply to test)
5. Time-based split (respect signup_date order)

### Models
- **LogisticRegression:** Default sklearn parameters, fitted on scaled features
- **GradientBoostingClassifier:** Default sklearn parameters, fitted on scaled features

### Experiment Design
- **Seeds:** 5
- **Metrics:** ROC-AUC (primary), F1, precision, recall, accuracy
- **Sanity checks:**
  - Baseline floor (majority class prediction)
  - Label shuffle test (verify no spurious leakage)
  - One check per seed ensures controls are repeatable

## Limitations & Risk

### Potential Leakage Surface
- ✓ **days_since_last_login** removed (encodes outcome timing)
- ✓ **Deduplication** prevents train-test contamination
- ✓ **Time-based split** respects signup_date order
- ⚠ **days_since_signup** derived from signup_date (safe, but could drift over time in production)

### Generalization
- Dataset is synthetic; real churn may have different feature relationships
- 4200 rows is small; variance estimates may be noisy
- Hyperparameters not tuned; this is a fair-baseline comparison, not an optimization race

## Raw Metrics

See `results/metrics.json` for per-seed results.
