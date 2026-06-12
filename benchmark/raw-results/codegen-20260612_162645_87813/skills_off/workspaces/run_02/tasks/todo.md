# Churn: GB vs LR

**Claim:** Does GradientBoostingClassifier outperform LogisticRegression at predicting `churned`?

**Variable:** model type only. Preprocessing, features, splits, tuning budget (defaults) held fixed.

## Leak surface (found before coding)
- [x] `account_status` = "closed" iff churned -> perfect target leak. DROP.
- [x] 200 exact duplicate rows -> would straddle a random split. DEDUP before split.
- [x] `signup_date` temporal, churn is forward-looking -> use time-based split, not random.
- [x] `customer_id` identifier -> DROP.

## Data contact policy
- Dedup, drop leak/id columns BEFORE any fit.
- Sort by signup_date; TimeSeriesSplit (expanding window) -> every test fold is strictly later than its train.
- StandardScaler fit on train fold only (inside Pipeline).

## Sanity checks
- [x] Baseline floor: DummyClassifier ROC AUC ~ 0.5.
- [x] Leakage ceiling: including account_status -> AUC ~ 1.0 (proves the leak; that's why it's dropped).
- [x] Label-shuffle: shuffled target -> AUC ~ 0.5.

## Metrics
- ROC AUC (primary, survives 27% imbalance), Average Precision (PR AUC), accuracy/F1 for context.
- 5 forward-looking folds -> mean +/- sd, n=5. Paired per-fold diffs for the comparison.

## Verify
- [x] pytest tests pass
- [x] run_experiment.py < 5 min, writes results/metrics.json + REPORT.md
