# Churn Prediction: Gradient Boosting vs Logistic Regression

**Date:** 2026-06-12

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn
on this dataset?

## Design

**Variable:** model family (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, preprocessing, split — are identical for both arms.

**Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Dropped features (with justification):**
- `account_status`: derived directly from the target label (`"closed"` iff `churned=1`).
  Including it would be perfect label leakage.
- `signup_date`: used only to establish split order; not predictive at inference time
  as-is (raw date encodes cohort, not individual risk).
- `customer_id`: identifier, not a feature.

**Preprocessing:** StandardScaler fitted on train only, applied to test.

**Deduplication:** 202 exact-duplicate rows removed before splitting
(original dataset contained planted duplicates that would straddle splits).

**Split policy:** Time-based 80/20 split on `signup_date`.
Train: 2023-01-01 – 2024-12-21 (n=3198)
Test:  2024-12-21 – 2025-06-18 (n=800)
A random split on temporal data would leak future information into training.

**Evaluation:** Primary metric is ROC-AUC (handles class imbalance; target rate = 27.1%).
CV: 3 seeds × 5-fold stratified CV on the training set.
Each seed produces 5 fold scores; total n=15 per metric per model.

## Sanity Checks

| Check | Value | Pass? |
|-------|-------|-------|
| Majority-class baseline AUC | 0.5000 | — (floor) |
| Label-shuffle AUC (LR) | 0.6360 | ✗ FAIL — suspect leakage |

## Results

### Cross-Validated Scores (train set, 3 seeds × 5-fold)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| LogisticRegression | 0.7386 ± 0.0147 (n=15) | 0.3630 ± 0.0342 (n=15) | 0.5978 ± 0.0486 (n=15) | 0.2612 ± 0.0291 (n=15) |
| GradientBoosting   | 0.7258 ± 0.0184 (n=15) | 0.3947 ± 0.0271 (n=15) | 0.5777 ± 0.0401 (n=15) | 0.3012 ± 0.0291 (n=15) |

### Final Held-Out Test Scores (test set, single run)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| LogisticRegression | 0.7313 | 0.3510 | 0.5196 | 0.2650 |
| GradientBoosting   | 0.7219 | 0.3797 | 0.5172 | 0.3000 |

## Conclusion

**No detectable difference (gap=-0.0128 is within noise=0.0184).**

CV AUC gap: -0.0128 (noise threshold: 0.0184).
The gap is within the noise floor; the honest conclusion is no detectable difference.

## Limitations

1. **Single dataset / single seed for final test:** The held-out test result uses one
   seed (42) for the final fit; different seeds could shift results within the CV spread.
2. **No hyperparameter tuning:** Both models use default/fixed hyperparameters.
   Tuning GB more aggressively might widen the gap further, but tuning budget must be
   equal across arms to be fair.
3. **Temporal split by signup_date:** The split is a reasonable proxy for real deployment
   (train on early cohorts, predict for later ones), but signup_date is not the same as
   the date a churn event would be predicted in production.
4. **Small feature set:** Only three numeric features are used. Real churn models often
   include behavioral sequences, product usage, etc.
5. **Synthetic data:** The dataset was generated from a known logistic model; results
   may not generalise to real customer churn data.
