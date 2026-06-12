# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

### Data
- **Raw rows:** 4200  →  **After deduplication:** 4000  (dropped 200 exact duplicates)
- **Overall churn rate:** 27.1%

### Rigor Steps Applied
| Issue | Action |
|-------|--------|
| `account_status` is "closed" iff churned — perfect target leak | Dropped before any modeling |
| 200 exact duplicate rows appended by the generator | `drop_duplicates()` before splitting |
| `signup_date` is temporal; random splits cause time leakage | Sort by date; earliest 80% → train, latest 20% → test |
| Scaler fitted on full data would leak test statistics | `StandardScaler` fit inside Pipeline per CV fold |

### Split
- **Method:** Time-based (sort by `signup_date`)
- **Cutoff date:** 2024-12-21 00:00:00
- **Train:** 3200 rows, churn rate 27.6%
- **Test:** 800 rows, churn rate 25.0%

### Evaluation Design
- **CV:** 5-fold stratified × 3 seeds = **15 estimates per model** (satisfies ≥3 requirement for variance claims)
- **Primary metric:** ROC-AUC (robust to class imbalance; summarises the full ranking curve)
- **Secondary metrics:** F1, precision, recall
- **Baseline:** Majority-class classifier (theoretical ROC-AUC = 0.5)
- **Features used:** `tenure_months`, `monthly_spend`, `support_tickets`
- **Test set touched:** once, at the end — no decisions were made after seeing test scores

## Results

### Sanity Check
Both models exceed the majority-class baseline (ROC-AUC = 0.5), confirming real signal is present.

### Cross-Validation on Training Set (n = 15 per model)

| Model | ROC-AUC mean ± sd | F1 mean ± sd | Precision mean ± sd | Recall mean ± sd |
|-------|-------------------|--------------|---------------------|------------------|
| Logistic Regression | 0.7376 ± 0.0207 | 0.3573 ± 0.0413 | 0.5975 ± 0.0710 | 0.2558 ± 0.0332 |
| Gradient Boosting   | 0.7232 ± 0.0206 | 0.3849 ± 0.0307 | 0.5824 ± 0.0425 | 0.2883 ± 0.0287 |

**CV AUC gap (GB − LR): -0.0145  (-0.7 SD)**

### Holdout Test Set (touched once)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| Logistic Regression | 0.7323 | 0.3510 | 0.5196 | 0.2650 |
| Gradient Boosting   | 0.7238 | 0.3950 | 0.5294 | 0.3150 |
| Majority-class baseline | 0.5000 | 0.0000 | — | — |

**Holdout AUC gap (GB − LR): -0.0085**

## Conclusion

**No detectable difference between the two models (CV AUC gap -0.0145, -0.7 SD — within noise).**

## Limitations

- **Single dataset, ~4000 rows:** Power to detect small differences is limited.
- **No hyperparameter tuning:** Default settings used for both models; tuned GB may change the relative gap.
- **Three features only:** After removing leaky columns, only `tenure_months`, `monthly_spend`, and `support_tickets` remain. Additional features might alter the comparison.
- **One temporal split:** A single cutoff means test-set variance is unobservable; walk-forward CV would give better estimates but at higher compute cost.
- **Deployment caveat:** Test performance reflects the held-out time window only; distribution shift beyond that window is unquantified.
