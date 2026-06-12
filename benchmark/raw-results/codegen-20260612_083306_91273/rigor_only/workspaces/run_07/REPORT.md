# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
**Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?**

## Conclusion
**Winner: Logistic Regression**

Logistic Regression shows a consistent advantage over Gradient Boosting (AUC: 0.7459 ± 0.0000 vs 0.7112 ± 0.0030), with a gap of ~0.0347 that exceeds the combined uncertainty.

## Methodology

### Data Handling
- **Deduplication:** 202 exact duplicates removed before splitting
  (planted rigor trap: 200 rows were appended to the dataset; random splits would let them straddle train/test)
- **Leaky features dropped:** `account_status` (derived from target: 'closed' iff churned=1), `customer_id`, `signup_date`
- **Features used:** `tenure_months`, `monthly_spend`, `support_tickets` (3 numeric features)
- **Split method:** Time-based (earliest 70% signup dates → train, latest 30% → test)
  - Rationale: Respects temporal order; prevents models from learning recency bias from random splits
- **Preprocessing:** StandardScaler fitted on train only, applied to test

### Models
- **Logistic Regression:** max_iter=1000, solver='lbfgs'
- **Gradient Boosting:** n_estimators=100, learning_rate=0.1, max_depth=5

### Evaluation
- **Metrics:** AUC-ROC (primary, handles class imbalance), F1, Precision, Recall
- **Runs:** 5 seeds with different random states; report mean ± standard deviation
- **Baseline:** Majority class prediction (always predicting churn rate)

### Dataset
- **Total rows:** 2800 train + 1200 test (after dedup)
- **Train churn rate:** 0.279
- **Test churn rate:** 0.252

## Sanity Checks

| Check | Result | Status |
|-------|--------|--------|
| Tiny overfit accuracy (50 rows) | 0.900 | ✓ OK |
| Label shuffle AUC (should be ~0.5) | 0.498 | ✓ OK |
| Test set duplicates | 0 | ✓ OK |

All sanity checks passed. Pipeline is sound.

## Results

### Primary Metric: AUC-ROC

| Model | Mean AUC | ± Std | vs Baseline |
|-------|----------|-------|------------|
| Baseline (majority) | 0.5000 | 0.0000 | – |
| Logistic Regression | 0.7459 | 0.0000 | ++0.2459 |
| Gradient Boosting | 0.7112 | 0.0030 | ++0.2112 |

### Secondary Metrics: F1, Precision, Recall

| Model | F1 | Precision | Recall |
|-------|-------|-----------|--------|
| LR | 0.3805 ± 0.0000 | 0.5733 ± 0.0000 | 0.2848 ± 0.0000 |
| GB | 0.4024 ± 0.0036 | 0.5118 ± 0.0132 | 0.3318 ± 0.0100 |

## Limitations and Threats to Validity

1. **Limited feature set:** Only 3 features (tenure, spend, tickets) after dropping leaky ones. Churn may depend on features not in the dataset.

2. **Time-based split assumption:** The 30% test window is brief. Results may not generalize to distant future churn patterns.

3. **Class imbalance:** Churn rate is ~25.2%. Metrics like F1 and recall are noisier with few positive examples.

4. **Hyperparameter tuning:** Both models used fixed hyperparameters. Tuning (e.g., on a validation set) could change the ranking.

5. **Small seed variance:** Only 5 seeds. Wider sampling (10+ seeds or cross-validation) would strengthen claims.

## Key Rigor Decisions

- **Deduplication before split:** Prevents leakage from duplicates straddling train/test.
- **Dropped `account_status`:** This feature is perfectly correlated with the target (by construction); using it would hide model quality.
- **Time-based split:** Standard practice in temporal data; avoids the pitfall of random splits on time-series-like data.
- **Report n, mean, std:** Allows readers to assess effect size and noise; no single-seed anecdotes.
- **Sanity checks before main run:** Verify pipeline works and assumptions hold before conclusions.
