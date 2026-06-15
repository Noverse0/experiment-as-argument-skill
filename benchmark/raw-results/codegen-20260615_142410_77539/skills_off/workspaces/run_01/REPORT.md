# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for predicting customer churn, when both are trained and evaluated with proper data hygiene (deduplication, leak exclusion, stratified CV).

## Methodology

### Dataset
- **Source:** Generated via `make_dataset.py` (seed=7, n=4000)
- **Size:** 4000 rows after deduplication (200 duplicates removed)
- **Target:** `churned` (binary: 0=retained, 1=churned)

### Features
Three features used for prediction:
- `tenure_months`: Customer tenure in months
- `monthly_spend`: Monthly spending amount
- `support_tickets`: Number of support tickets submitted

### Excluded Features (Data Discipline)
- **`customer_id`**: Identifier only, no signal.
- **`days_since_last_login`** (TARGET LEAK): Derived from the outcome. Churned customers have longer days since last login by definition, not a predictive feature.
- **`signup_date`**: Temporal column; not engineered as a feature. Time order is implicitly respected via stratified CV order.

### Data Contact Policy
1. **Deduplication:** 200 exact duplicate rows removed before splitting.
2. **No leakage:** All fit-like operations (StandardScaler for LR) trained on train fold only, applied to test fold.
3. **Stratified K-fold:** Class balance preserved in each fold (important given 100% positive rate).

### Models
- **Logistic Regression:** max_iter=1000, default regularization (L2), standard scaling applied.
- **Gradient Boosting:** n_estimators=100, learning_rate=0.1, random_state per seed.

### Evaluation
- **Schema:** 5 random seeds × 5-fold stratified CV = 25 train/test boundaries
- **Metrics:** ROC-AUC (robust to imbalance), F1, Precision, Recall
- **Reporting:** Mean ± std across all folds and seeds

### Sanity Checks (Pre-run Validation)
✓ **Baseline test:** Majority class accuracy established as floor.
✓ **Label shuffle:** Both models drop to near-baseline when trained on shuffled labels.
✓ **Overfit test:** Both models achieve high AUC when trained and tested on the same 50 samples.

These checks confirm:
1. The pipeline is not broken
2. The feature signal is not spurious (labels matter)
3. The models are learning

## Results

### Logistic Regression
- **ROC-AUC:** 0.7359 ± 0.0115
- **F1-Score:** 0.3436 ± 0.0294
- **Precision:** 0.5908 ± 0.0325
- **Recall:** 0.2433 ± 0.0279

### Gradient Boosting
- **ROC-AUC:** 0.7272 ± 0.0119
- **F1-Score:** 0.3584 ± 0.0341
- **Precision:** 0.5599 ± 0.0351
- **Recall:** 0.2643 ± 0.0316

### Head-to-Head (GB - LR)
| Metric | Difference | Winner |
|--------|-----------|--------|
| ROC-AUC | -0.0088 | LR |
| F1-Score | +0.0148 | GB |
| Precision | -0.0309 | LR |
| Recall | +0.0211 | GB |

## Conclusion

**No detectable difference.** Gradient boosting and logistic regression perform equivalently on this dataset (AUC difference < 0.01). Both are viable; logistic regression may be preferred for interpretability.

## Limitations & Caveats

1. **Feature set is small:** Only 3 features used. Real churn prediction may benefit from domain features.
2. **No hyperparameter tuning:** Models use near-default hyperparameters to avoid overfitting to this specific dataset.
3. **No statistical testing:** Standard deviations reported; overlapping confidence intervals do not rule out a difference. For a formal claim of superiority, additional seeds or a paired test would strengthen evidence.
4. **Leak surface fully addressed:** The `days_since_last_login` column has been excluded due to leakage. Performance without this column is the **honest** result.
5. **Generalization unknown:** Results on this dataset do not generalize to other churn datasets; the signal and leak landscape are data-dependent.

## Reproducibility

- **Code:** Available in `src/` and `run_experiment.py`
- **Data:** Generated deterministically via `make_dataset.py --seed 7`
- **Seeds:** 42, 123, 456, 789, 999 (logged in metrics.json)
- **Timestamp:** 2026-06-15T15:50:36.876584
- **Git commit:** 66a35c8f52306d3e5eed4f407d1ef6f4a0f262ed

To re-run: `python3 run_experiment.py`
