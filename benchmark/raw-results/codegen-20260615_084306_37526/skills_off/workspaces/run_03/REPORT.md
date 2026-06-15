# Churn Prediction Experiment Report

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Variable:** Model class (scikit-learn `LogisticRegression` vs `GradientBoostingClassifier`, default hyperparameters). All other choices are held fixed across both arms.

### Data preparation

| Step | Action | Reason |
|------|--------|--------|
| Deduplication | Removed 200 exact-duplicate rows | Duplicates straddling the train/test boundary inflate held-out performance |
| Temporal sort | Sorted by `signup_date` ascending | Enforces chronological integrity for the split |
| Time-based split | First 80% → train (3200 rows), last 20% → test (800 rows) | Random splits on temporal data allow future-signup customers to inform past-signup predictions (a form of leakage) |

### Feature selection

| Feature | Status | Reason |
|---------|--------|--------|
| `tenure_months` | **Used** | Causal: shorter tenure → higher churn probability |
| `monthly_spend` | **Used** | Causal: higher spend → slightly higher churn probability |
| `support_tickets` | **Used** | Causal: more tickets → higher churn probability |
| `days_since_last_login` | **Dropped — TARGET LEAK** | Churned customers have stopped logging in by definition. This value is derived from (i.e., causally downstream of) the churn outcome. It encodes the label, not a cause, and would be unavailable at real prediction time. |
| `customer_id` | Dropped | Row identifier; no predictive value |
| `signup_date` | Dropped as feature | Used only for temporal split ordering; including it as a numeric feature risks encoding time-of-split information |

### Class balance

- Train churn rate: 27.6%
- Test churn rate: 25.0%

### Evaluation protocol

- **Primary metric:** ROC-AUC (robust to class imbalance; threshold-independent)
- **Secondary metrics:** Average Precision, F1 (threshold = 0.5)
- **Variance estimation:** 3 random seeds × 5-fold StratifiedKFold = 15 CV fits per model on the training set
- **Preprocessing:** `StandardScaler` fitted on each CV train fold only, applied to the corresponding val fold — no information from the validation set enters the scaler
- **Final evaluation:** Models trained on all train data; test set touched exactly once, after all design decisions were fixed

## Results

### Cross-Validation Performance (training set, 15 folds per model)

| Model | ROC-AUC (mean ± std) | Avg Precision (mean ± std) | F1 (mean ± std) |
|-------|---------------------|---------------------------|-----------------|
| LogisticRegression | 0.7377 ± 0.0241 | 0.5114 ± 0.0343 | 0.3566 ± 0.0470 |
| GradientBoosting | 0.7278 ± 0.0240 | 0.4931 ± 0.0354 | 0.3833 ± 0.0360 |

AUC gap (GB − LR): -0.0099 | Pooled CV std: 0.0241

### Held-Out Test Set Performance (n = 800, touched once)

| Model | ROC-AUC | Avg Precision | F1 |
|-------|---------|--------------|-----|
| LogisticRegression | 0.7323 | 0.4922 | 0.3510 |
| GradientBoosting | 0.7238 | 0.4812 | 0.3950 |

## Conclusion

**No statistically meaningful difference detected between the two models.**

The AUC gap (0.0099) is within the pooled CV standard deviation (0.0241). With these defaults and this dataset, gradient boosting does not reliably outperform logistic regression.

This outcome is consistent with the data-generating process: the true signal is linear in the log-odds (`logit = −1.2 − 0.03·tenure + 0.01·spend + 0.45·tickets`), a structure for which logistic regression is the correctly-specified model. Gradient boosting's additional capacity for nonlinear interactions does not provide a systematic advantage over a well-matched linear model on this dataset.

## Limitations

1. **No hyperparameter tuning:** Both models use scikit-learn defaults. A properly tuned gradient boosting model (lower learning rate, more trees, subsampling) might show a larger advantage; a systematic comparison would require an inner CV tuning loop for both arms.
2. **Single synthetic dataset:** Results are specific to this data-generating process. Real churn datasets often have nonlinear feature interactions that favor tree ensembles.
3. **StratifiedKFold within temporal training data:** The CV folds do not strictly preserve chronological order within the training portion. A `TimeSeriesSplit` would be more conservative. The held-out test set (temporally after all training data) provides a clean final evaluation.
4. **15 CV observations per model:** Provides reasonable but not definitive variance estimates; overlapping confidence intervals are the honest conclusion, not a significant winner claim.
5. **Synthetic linear signal:** The true generative model is additive and linear in the log-odds, which structurally favors logistic regression. A gradient boosting advantage might emerge with nonlinear or interaction-heavy real-world data.
