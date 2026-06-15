# Churn Prediction Experiment — Results

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Verdict: LOGISTIC REGRESSION WINS

LR outperforms GBM by 0.0182 AUC on average (noise ±0.0150). The gap exceeds fold variance.

---

## Methodology

### Dataset
- Rows after deduplication: **4000** (200 planted exact duplicates removed)
- Churn rate: **27.1%** (1082 churned / 2918 retained)

### Feature Selection
**Used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Dropped and why:**
| Column | Reason |
|---|---|
| `customer_id` | Identifier — carries no signal |
| `signup_date` | Used for temporal ordering only; not a predictive feature |
| `days_since_last_login` | **Target leak** — post-outcome field. A churned customer has, by definition, already stopped logging in when churn is recorded. Including it would inflate AUC without providing real predictive power on unseen future customers. |

### Evaluation
- **Split strategy:** `TimeSeriesSplit` with `n_splits=5`, applied after sorting by `signup_date`.  Each fold trains on earlier-signup cohorts and validates on later ones, mimicking a real deployment where we always predict for customers who joined after our training window.
- **Why not random split:** With temporal data, random splits allow future information to leak into training. Straddling duplicates (now removed) would further inflate metrics.
- **Primary metric:** ROC-AUC (threshold-free; appropriate for imbalanced binary classification).
- **Models:** `LogisticRegression(max_iter=1000)` with `StandardScaler`; `GradientBoostingClassifier(n_estimators=100, max_depth=3, lr=0.1, subsample=0.8)`.
- **Baseline floor:** Majority-class predictor → AUC = 0.500.

---

## Results

| Model | AUC (mean ± sd) | F1 (mean ± sd) | Accuracy (mean ± sd) |
|---|---|---|---|
| LogisticRegression | 0.7329 ± 0.0252 | 0.3694 ± 0.0401 | 0.7489 ± 0.0230 |
| GradientBoosting   | 0.7147 ± 0.0213 | 0.4040 ± 0.0329 | 0.7387 ± 0.0213 |

**Per-fold AUC values (n=5):**

| Fold | LogisticRegression | GradientBoosting | Δ (GBM−LR) |
|------|---|---|---|
| 1 | 0.7338 | 0.6986 | -0.0352 |
| 2 | 0.7378 | 0.7096 | -0.0282 |
| 3 | 0.6951 | 0.6927 | -0.0024 |
| 4 | 0.7659 | 0.7435 | -0.0224 |
| 5 | 0.7318 | 0.7291 | -0.0026 |

**Mean AUC gap (GBM − LR):** -0.0182 ± 0.0150

Both models substantially exceed the baseline AUC of 0.500.

---

## Limitations and Validity Threats

1. **Small feature set:** Only three features survived the leak audit. More legitimate features (e.g., product usage counts) could shift the relative advantage.
2. **Single dataset / single seed:** The dataset is synthetically generated. Results may not generalise to production churn data.
3. **No hyperparameter search under shared budget:** LR and GBM were compared with default/reasonable hyperparameters. A proper comparison would equalise the tuning budget across arms.
4. **Temporal cohort effects:** Later signup cohorts may behave differently from earlier ones (distribution shift), which could interact with the temporal CV design.
5. **`days_since_last_login` excluded:** While necessary for validity, this removes what would be a strong operational signal in real deployments (where it would be recorded before the prediction window closes).
