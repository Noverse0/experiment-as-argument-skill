# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Conclusion
**No statistically significant difference** between models on this dataset.

**AUC Gap (GB - LR):** -0.0044 ± 0.0084 (95% CI: [-0.0208, 0.0121])

---

## Results

### Logistic Regression
- **AUC:** 0.7243 ± 0.0159
  - Runs: ['0.7399', '0.7306', '0.7025']
- **F1:** 0.3443 ± 0.0040
  - Runs: ['0.3419', '0.3498', '0.3410']

### Gradient Boosting
- **AUC:** 0.7200 ± 0.0196
  - Runs: ['0.7403', '0.7261', '0.6935']
- **F1:** 0.3493 ± 0.0226
  - Runs: ['0.3812', '0.3322', '0.3344']

---

## Methodology

### Data Preparation
- **Original dataset:** 4,200 rows (4,000 observations + 200 exact duplicates)
- **After deduplication:** 3,897 rows
- **Train/test split:** 80% / 20% stratified on target to maintain class balance
- **Preprocessing:** StandardScaler fitted on train only, applied to both sets

### Features Used
Three honest causal signals retained:
- `tenure_months`: Customer tenure (months as customer)
- `monthly_spend`: Average monthly spending
- `support_tickets`: Number of support interactions

### Excluded Columns & Rationale
- `customer_id`: Unique identifier with no predictive signal
- `signup_date`: Temporal feature; documented as limitation of random split
- `days_since_last_login`: **TARGET LEAKAGE** — churned customers, by definition, have higher values recorded at/after the outcome. Not input available at prediction time in a production setting.

### Model Configuration
- **LogisticRegression:** L2 regularization (default C=1.0), max_iter=1000
- **GradientBoostingClassifier:** Default scikit-learn parameters

### Evaluation Metrics
- **AUC-ROC:** Primary metric (robust to class imbalance)
- **F1-Score:** Secondary metric
- **Accuracy, Precision, Recall:** Reported for completeness

### Runs & Variance
- **Repeated across 3 random seeds:** [42, 123, 456]
- **Class balance maintained:** Train churn rate ≈ test churn rate across seeds
- **Metrics reported as:** mean ± std over runs

---

## Sanity Checks

### 1. Baseline Floor
Both models must outperform the majority-class baseline (always predict "no churn").
✓ **PASSED:** Both models achieve AUC >> 0.5

### 2. Label Shuffle
With shuffled training labels, model performance must collapse to the baseline floor (verifies no spurious leakage).
✓ **PASSED:** Shuffled label AUC ≈ 0.5

### 3. Leakage Ceiling
When `days_since_last_login` is included, we expect suspiciously high AUC (confirms the leak is real and potent).
- LR with leak: 0.9705
- GB with leak: 0.9670
✓ **CONFIRMED:** Models with leakage achieve much higher AUC than honest features alone.

---

## Limitations & Caveats

1. **Temporal Structure Ignored:** `signup_date` is in the dataset but not used; a time-based split would be more realistic for a forward-looking churn prediction task.

2. **Duplicates:** The 200 exact duplicate rows (removed before splitting) were added to the dataset intentionally. In a production setting, duplicates straddling train/test would be a real concern.

3. **Real-time feasibility:** `days_since_last_login` is a post-outcome leak — it is not known at prediction time. The honest features (tenure, spend, support) would be the only information available in a real system.

4. **Class imbalance:** Overall churn rate is ~27%, which is moderate. AUC is appropriate; accuracy alone would be misleading.

5. **Generalization:** Results are specific to this dataset and random seed choice. Conclusions should not be extrapolated to other churn datasets without validation.

---

## Artifacts
- `results/metrics.json`: Full results in machine-readable format
- `REPORT.md`: This report

---

*Experiment completed on 2026-06-15 using scikit-learn on CPU.*
