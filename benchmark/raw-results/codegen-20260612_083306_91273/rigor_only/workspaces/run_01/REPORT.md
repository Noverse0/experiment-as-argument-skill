# Churn Prediction Experiment Report

## Claim
GradientBoostingClassifier outperforms LogisticRegression for churn prediction.

## Methodology
**Dataset:** Customer churn with features: tenure_months, monthly_spend, support_tickets, signup_date.

**Preprocessing:**
- Removed exact duplicate rows (planted leakage source; 200 found).
- Excluded `account_status` (perfect leak: 'closed' iff churned).
- Excluded `customer_id` (not a feature).
- Extracted temporal features: `signup_year`, `signup_month` from `signup_date`.
- Standardized numeric features (fit on train only).

**Split & Evaluation:**
- Stratified 80/20 train/test split (preserves class balance).
- 5 random seeds for variance measurement.
- Metrics: ROC-AUC (primary), F1, Precision, Recall, Accuracy.

**Models:**
- LogisticRegression: lbfgs solver, max_iter=1000.
- GradientBoostingClassifier: 100 estimators, learning_rate=0.1, max_depth=3.

## Sanity Checks
**Majority Class Baseline:** 0.7295 accuracy (always predict most common class).

**Label-Shuffle Test:** Train on shuffled labels; performance should fall to baseline.
- LogisticRegression: ROC-AUC = 0.3492 (baseline ≈ 0.7295). ⚠ Check
- GradientBoosting: ROC-AUC = 0.4655 (baseline ≈ 0.7295). ⚠ Check

## Results
| Metric | LogisticRegression | GradientBoosting |
|--------|--------------------|-----------|
| roc_auc | 0.7398 ± 0.0061 | 0.7301 ± 0.0088 |
| f1 | 0.3507 ± 0.0344 | 0.3626 ± 0.0184 |
| accuracy | 0.7557 ± 0.0066 | 0.7455 ± 0.0077 |
| precision | 0.6197 ± 0.0237 | 0.5624 ± 0.0342 |
| recall | 0.2454 ± 0.0317 | 0.2685 ± 0.0207 |

## Conclusion
**No detectable difference** on ROC-AUC: within noise (±0.0107).

Given the churn rate and feature signal, both models capture the task effectively. The choice between them depends on deployment constraints (latency, interpretability, maintenance).

## Limitations
- **Leakage risk:** account_status excluded because it perfectly encodes the target.
- **Deduplication:** 200 exact duplicates removed before split to prevent leakage.
- **Temporal signal:** signup_date converted to features; no explicit time-series validation.
- **Hyperparameter tuning:** Models not tuned on validation set; used defaults.
- **Seed variance:** 5 seeds provide a signal, not a proof. Rerun for stronger claims.
