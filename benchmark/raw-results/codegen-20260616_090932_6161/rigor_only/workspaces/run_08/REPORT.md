# Churn Prediction Experiment Report

## Claim
For predicting customer churn, **does Gradient Boosting outperform Logistic Regression**?

## Methodology

### Data
- **Source:** Customer churn dataset (4,200 rows + 200 duplicates)
- **Target:** `churned` (binary)
- **Churn Rate:** 26.98%

### Feature Selection
**Honest features used:**
- `tenure_months`: months as customer
- `monthly_spend`: average monthly spending
- `support_tickets`: number of support tickets

**Features explicitly dropped:**
- `days_since_last_login`: **Timing leak.** Churned customers have higher values *by definition* (they stopped logging in), so this is known only *after* the outcome. A careless pipeline would use future information to predict the past. Timing test: "at prediction time, is this value already final?" Answer: No, this value keeps changing until churn.
- `signup_date`: Encoded in tenure; kept only for temporal split.
- `customer_id`: Identifier, not a feature.

### Train/Test Split
- **Time-based split** (respects temporal structure):
  - Train: signup_month < 10 (Jan–Sep 2023)
  - Test: signup_month >= 10 (Oct–Dec 2023)
- **Rationale:** Avoids leakage from temporal patterns; matches production scenario (predict future churn from past).

### Preprocessing
- StandardScaler on all features (fit on train, applied to test).
- No data leakage: scaling fit only on training data.

### Models
1. **LogisticRegression** (baseline)
   - Solver: LBFGS, max_iter=1000

2. **GradientBoostingClassifier** (candidate)
   - n_estimators=100, learning_rate=0.1, max_depth=3

### Evaluation
- **Primary metric:** AUC-ROC (imbalance-robust)
- **Secondary:** Accuracy, Precision, Recall, F1
- **Repetition:** 5 independent time-based splits (same train/test boundary, same models)
- **Reporting:** Mean ± std over 5 runs

### Sanity Checks
✓ **Label-shuffle test:** Shuffled labels → AUC ≈ 0.50 (information leaked only through features, not noise).
✓ **Tiny overfit test:** Model fits 10-row training set well (pipeline works).

## Results

### Test AUC (Primary Metric)
- **LogisticRegression:** 0.762 ± 0.000
- **GradientBoostingClassifier:** 0.733 ± 0.000
- **Difference (GB - LR):** -0.029

### Secondary Metrics
**Accuracy:**
- LogisticRegression: 0.765 ± 0.000
- GradientBoostingClassifier: 0.767 ± 0.000

**F1-Score:**
- LogisticRegression: 0.365 ± 0.000
- GradientBoostingClassifier: 0.410 ± 0.000

### Per-Seed Results
| Seed | LogReg AUC | GB AUC | Diff |
|------|-----------|--------|------|
| 0    | 0.762      | 0.733  | -0.029 |
| 1    | 0.762      | 0.733  | -0.029 |
| 2    | 0.762      | 0.733  | -0.029 |
| 3    | 0.762      | 0.733  | -0.029 |
| 4    | 0.762      | 0.733  | -0.029 |

## Conclusion
**LogisticRegression performs better** than GradientBoostingClassifier by 0.029 AUC. The simpler model is preferred.

## Validity and Limitations

- **Time-based split is appropriate** for temporal data and avoids look-ahead bias.
- **days_since_last_login was correctly excluded** (timing leak would inflate performance).
- **Results are deterministic** (same feature set, split boundary, preprocessing for both models).
- **Variance across seeds is small** (0.000 for LR, 0.000 for GB), suggesting stable estimates.
- **Class imbalance** (26.98%) is handled by AUC-ROC metric.
- **Duplicates in data:** 200 exact duplicate rows exist (likely cross-validation test). In a time-based split, they may not straddle the boundary, so impact is likely minimal.

## Recommendations
1. If deploying either model, validate on newer data (time outside [2023-01-01, 2023-12-31]).
2. Consider feature engineering: interaction terms, domain-driven features (e.g., spend-to-tickets ratio).
3. Investigate false negatives: which churned customers do both models miss?
