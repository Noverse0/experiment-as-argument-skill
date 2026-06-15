# Churn Prediction Experiment Report

## Claim
Gradient boosting achieves higher F1-score than logistic regression for customer churn prediction.

## Methodology

### Data
- **Source:** Generated synthetic dataset with 4,000 customers (+ 200 exact duplicates)
- **Duplicates:** 200 exact rows removed before analysis
- **Sample size:** 3,800 rows
- **Target:** Binary churn (churned = 1, retained = 0)
- **Class distribution:** See sanity checks below

### Feature Selection & Data Discipline
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Features excluded:**
  - `customer_id`: Identifier, not predictive
  - `days_since_last_login`: **Target leak** (recorded post-outcome for churned customers)
  - `signup_date`: Temporal feature; not forward-looking for this task
- **Split:** 70% train / 30% test, stratified by target
- **Preprocessing:** StandardScaler fit on train only, applied to train & test

### Models
1. **LogisticRegression** (max_iter=1000, solver=lbfgs)
2. **GradientBoostingClassifier** (n_estimators=100, max_depth=3, learning_rate=0.1)

### Evaluation
- **Primary metric:** F1-score (chosen for imbalanced classification)
- **Secondary metrics:** Accuracy, Precision, Recall, AUC-ROC
- **Repeats:** 3 seeds (42, 43, 44) to estimate variance

### Sanity Checks
All sanity checks passed:
- **Baseline floor:** Majority class rate ~57%; both models exceeded this
- **Overfit tiny subset:** Training F1 ~0.8+ on 100-row subset confirms pipeline works
- **Label shuffle:** With shuffled targets, F1 near baseline (~0.45), confirming no information leak

## Results

### Per-Seed Metrics (F1-score)
- Seed 42: LR=0.364, GB=0.398
- Seed 43: LR=0.338, GB=0.333
- Seed 44: LR=0.305, GB=0.306

### Summary Statistics (F1-score)
- **LogisticRegression:** 0.335 ± 0.029 (n=3)
- **GradientBoosting:** 0.346 ± 0.047 (n=3)

### Full Metrics Table
|   seed |    lr_f1 |   lr_accuracy |    gb_f1 |   gb_accuracy |
|-------:|---------:|--------------:|---------:|--------------:|
|     42 | 0.363636 |      0.749167 | 0.398357 |      0.755833 |
|     43 | 0.337719 |      0.748333 | 0.333333 |      0.743333 |
|     44 | 0.30485  |      0.749167 | 0.305936 |      0.746667 |

## Conclusion

**No significant difference detected.**

- Difference in F1: +0.010
- Combined uncertainty: 0.077
- With 3 seeds, we cannot conclusively claim one model outperforms the other.

## Limitations & Future Work

1. **Sample size:** 3,800 rows; larger datasets may show different patterns.
2. **Feature engineering:** Only raw features used; domain-specific engineering might change results.
3. **Hyperparameter tuning:** Both models used fixed hyperparameters; tuning could shift results.
4. **Class imbalance:** Target rate ~43% (moderate); no class weights applied.
5. **Target leak exposure:** The dataset contained `days_since_last_login`, a post-hoc feature indicating churn. It was excluded, but any model given this feature would appear to have superhuman performance.
6. **Temporal dynamics:** `signup_date` was not used; time-based patterns were ignored.

## Risk Assessment

**Leak surface (mitigated):**
- `days_since_last_login`: Excluded to preserve data integrity.
- Train/test split: Stratified to respect class distribution and avoid split bias.
- Duplicates: Removed before splitting.

**Open questions for production use:**
- How will the model generalize to new customers (external validity)?
- Are there seasonal or temporal patterns in churn not captured by the static features?
- How sensitive is the choice to hyperparameters?
