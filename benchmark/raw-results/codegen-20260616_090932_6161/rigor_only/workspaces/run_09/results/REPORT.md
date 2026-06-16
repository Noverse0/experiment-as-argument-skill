# Churn Prediction Experiment Report

## Claim
GradientBoostingClassifier achieves superior ROC-AUC and recall compared to LogisticRegression for customer churn prediction.

## Methodology

### Data Preparation
- **Dataset:** churn.csv (4,200 samples)
- **Features Used:** tenure_months, monthly_spend, support_tickets
- **Target:** churned (binary, 73.0% positive rate)

### Feature Justification
- `tenure_months`: Fixed historical feature, no leakage
- `monthly_spend`: Recent spending behavior, pre-churn
- `support_tickets`: Historical count of support interactions

### Dropped Features (Leakage Analysis)
- `customer_id`: No predictive signal
- `signup_date`: Redundant with tenure_months
- `days_since_last_login`: **TIMING LEAK** — If a customer churned, their last login date is fixed in the past. At prediction time, this value encodes the churn outcome. Fails the timing test: "Is this value already final at prediction time?" Yes — churn determines the login date.

### Experimental Design
- **Splits:** Random 70/30 train-test split, stratified by target
- **Preprocessing:** StandardScaler fitted on training set only, applied to test
- **Repetitions:** 5 random seeds (40, 41, 42, 43, 44)
- **Reporting:** Mean ± standard deviation across seeds

### Sanity Checks (Passed)
1. **Baseline Floor:** Both models exceed majority-class accuracy (73.0%)
2. **Label-Shuffle Test:** With shuffled labels, both models revert to baseline
3. **Determinism:** Identical runs with same seed produce identical metrics

### Models Compared
- **LogisticRegression:** max_iter=1000, default regularization
- **GradientBoostingClassifier:** 100 estimators, depth=3, learning_rate=0.1

### Metrics
- **ROC-AUC:** Primary metric (handles imbalance well)
- **Recall:** Important for churn (catch as many churners as possible)
- **Precision:** Cost of false positives
- **F1, Balanced Accuracy, Accuracy:** Supporting metrics

## Results

### LogisticRegression
- **accuracy:** 0.7522 ± 0.0095
- **balanced_accuracy:** 0.5887 ± 0.0115
- **f1:** 0.3372 ± 0.0243
- **precision:** 0.6069 ± 0.0490
- **recall:** 0.2335 ± 0.0170
- **roc_auc:** 0.7364 ± 0.0068

### GradientBoostingClassifier
- **accuracy:** 0.7490 ± 0.0082
- **balanced_accuracy:** 0.5932 ± 0.0136
- **f1:** 0.3532 ± 0.0318
- **precision:** 0.5805 ± 0.0353
- **recall:** 0.2547 ± 0.0307
- **roc_auc:** 0.7352 ± 0.0110

## Conclusion

**Primary Metric (ROC-AUC):**
- LogisticRegression: 0.7364 ± 0.0068
- GradientBoostingClassifier: 0.7352 ± 0.0110
- Gap: 0.0012 (standard error: 0.0129)

**Winner:** **No significant difference detected.** The gap is within noise.

## Limitations
1. **Dataset Size:** 4,200 samples is modest; results may not generalize to larger populations
2. **Feature Engineering:** Limited to three numeric features; domain-specific features (seasonal patterns, contract type) could improve both models
3. **Hyperparameter Tuning:** No hyperparameter search performed; both models use defaults
4. **Temporal Aspect:** Random split ignores temporal ordering; a time-based split would be more realistic for churn prediction
5. **Class Imbalance:** Target distribution not heavily imbalanced; results may differ on more skewed datasets

## Recommendations
- If GradientBoostingClassifier wins: Deploy it for production churn scoring
- If tied: Choose LogisticRegression for interpretability and training speed
- Future: Conduct hyperparameter search on larger evaluation set (e.g., nested CV) and measure feature importance
