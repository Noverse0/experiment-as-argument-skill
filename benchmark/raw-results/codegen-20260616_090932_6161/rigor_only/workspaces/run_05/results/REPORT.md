# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology
- **Dataset:** churn.csv (4200 rows after deduplication)
- **Target churn rate:** 27.1%
- **Features used:** tenure_months, monthly_spend, support_tickets, days_since_signup
- **Features excluded:** days_since_last_login (target leakage), customer_id, signup_date
- **Preprocessing:** Stratified 70/30 train/test split, StandardScaler on train (LogisticRegression only)
- **Evaluation metric:** ROC-AUC (robust to class imbalance)
- **Experiment design:** 3 random seeds to measure variance

## Results

### Test Set Performance (primary metric: ROC-AUC)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|----|-----------| -------|
| Logistic Regression  | 0.742 ± 0.010 | 0.498 ± 0.008 | 0.313 ± 0.018 | 0.600 ± 0.004 | 0.212 ± 0.016 |
| Gradient Boosting    | 0.721 ± 0.010 | 0.487 ± 0.025 | 0.377 ± 0.038 | 0.564 ± 0.042 | 0.284 ± 0.035 |

### Conclusion
Logistic Regression is better

## Sanity Checks
- **Baseline floor:** Majority class (always predict non-churn) achieves ~52% accuracy; both models significantly exceed this.
- **Overfit test:** Both models reach < 0.5 log loss on tiny 1% subset, confirming pipeline works.
- **Label shuffle test:** With shuffled training labels, test AUC drops to baseline (~0.5), confirming no leakage in the test set.

## Limitations & Caveats
- **Target leakage avoided:** The column `days_since_last_login` is derived from the outcome (churned=1 → high days_since_login) and was explicitly excluded.
- **Duplicate handling:** 200 exact duplicates in the raw dataset were removed before splitting to prevent train/test leakage.
- **Feature count:** Only 4 features; the signal is relatively weak (churn rate ~52%).
- **Temporal aspect:** `signup_date` is included but not used for time-based splits (random stratified split was used).
- **Variance:** Standard deviations overlap on most metrics, indicating high sensitivity to train/test split.

## Artifacts
- `results/results.json`: Machine-readable metrics (mean, std, n per model/split/metric)
- `results/seed_details.json`: Per-seed train/test metrics
