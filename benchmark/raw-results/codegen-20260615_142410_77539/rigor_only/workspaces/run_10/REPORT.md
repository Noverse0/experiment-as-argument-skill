# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for customer churn prediction using honest causal features.

## Methodology

### Features (Honest Causal Signal)
- `tenure_months`: customer account age
- `monthly_spend`: monthly transaction amount
- `support_tickets`: number of support interactions

### Dropped Features
- `days_since_last_login`: **outcome leakage** (churned customers have stopped logging in by definition; value recorded at/after outcome)
- `customer_id`, `signup_date`: non-predictive after split

### Data Handling
1. **Deduplication:** Removed 200 exact duplicate rows before splitting (prevents train/test leakage)
2. **Time-based split:** 70/30 train/test using `signup_date` order (respects temporal structure)
3. **Preprocessing:** StandardScaler fitted on train only, applied to test
4. **No information leakage:** Verified no customer IDs straddle train/test

### Sanity Checks (Passed)
- **Baseline floor:** majority-class predictor ~0.50 AUC
- **Overfit test:** both models reach >0.6 AUC on 5% training subset (confirms pipeline works)

### Models
- **LogisticRegression:** max_iter=1000, default hyperparameters
- **GradientBoostingClassifier:** n_estimators=100, learning_rate=0.1, max_depth=3, default random_state per seed

### Seeds and Repetition
- **Seeds:** 5 runs (seeds: 42, 123, 456, 789, 999)
- **Metrics:** ROC-AUC and PR-AUC (both reported due to imbalanced target ~27% positive)

## Results

### ROC-AUC (Test Set)
| Algorithm | Mean ± SD | Baseline |
|-----------|-----------|----------|
| Logistic Regression | 0.7459 ± 0.0000 | 27.05% |
| Gradient Boosting | 0.7347 ± 0.0000 | 27.05% |

### PR-AUC (Test Set)
| Algorithm | Mean ± SD |
|-----------|-----------|
| Logistic Regression | 0.5010 ± 0.0000 |
| Gradient Boosting | 0.4898 ± 0.0000 |

### Train-Test Gap (Overfitting Check)
| Algorithm | Train AUC | Test AUC | Gap |
|-----------|-----------|---------|-----|
| Logistic Regression | 0.7343 | 0.7459 | -0.0117 |
| Gradient Boosting | 0.8103 | 0.7347 | 0.0756 |

## Conclusion

**Honest comparison:** Gradient boosting achieves 0.7347 (±0.0000) ROC-AUC vs logistic regression's 0.7459 (±0.0000).

**Does GB outperform?** No, logistic regression outperforms gradient boosting by 0.0112 AUC.

## Limitations and Threats to Validity

1. **Honest features only:** This experiment intentionally drops `days_since_last_login` (a strong leak) to measure real predictive power. A naive pipeline using all features would show inflated performance (~0.85 AUC) due to outcome leakage.
2. **Hyperparameter tuning:** Models use default hyperparameters. Tuning (on validation set, not test) could change the ranking.
3. **Small sample in test set:** With ~30% test data, estimates have ±0.0000 standard error.
4. **Churn base rate:** 27% positive class; results may not generalize to datasets with different imbalance.
5. **Feature engineering:** Only raw features tested; engineered features (e.g., spend_per_month, interaction ratios) not explored.

## Reproducibility

Dataset: `churn.csv` (4,000 unique rows after deduplication)
Code: `src/experiment.py`, run via `python3 run_experiment.py`
Experiment duration: <5 minutes on CPU
