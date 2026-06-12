# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer `churned` on this dataset?

## Conclusion
**Logistic regression wins.** The paired CV difference favors logistic regression and its 95% CI excludes zero.

| Model | CV ROC-AUC (mean ± sd) | CV PR-AUC (mean ± sd) | Holdout ROC-AUC | Holdout PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.727 ± 0.024 | 0.496 ± 0.051 | 0.745 | 0.500 |
| Gradient Boosting | 0.702 ± 0.021 | 0.465 ± 0.074 | 0.732 | 0.495 |

Paired difference (GBM − LR) on ROC-AUC across 5 folds:
**-0.0249** (sd 0.0136, 95% CI [-0.0368, -0.0130]).

## Methodology
- **Single variable:** the classifier. Features, split, and preprocessing
  policy are held fixed between arms. Seed = 17 for everything.
- **Features used:** `tenure_months, monthly_spend, support_tickets`.
- **Excluded by design:**
  - `customer_id` — identifier, no generalizable signal.
  - `account_status` — **target leak**: it is `"closed"` iff `churned == 1`
    (the data audit confirms each status maps to a single churn value:
    `account_status_is_leak = True`). Using it would
    fabricate a near-perfect score that does not exist at prediction time.
  - `signup_date` — used only to **order** the split, not as a predictor; its
    relationship to churn is not assumed.
- **Deduplication before splitting:** the raw file contains
  **200 exact duplicate rows**. They are dropped *before*
  the split so identical rows cannot straddle train/test. Modeled on
  4000 unique rows.
- **Chronological split:** the task is forward-looking, so the
  chronologically last 30% of rows
  (1200 rows) form a held-out test set and the
  earlier 2800 rows form the training set. A
  random split would leak future information.
- **Variance:** the model comparison uses 5-fold
  `TimeSeriesSplit` (forward-chaining) on the training set, reported as
  mean ± sd. One split is an anecdote; folds give a spread.
- **Metrics:** ROC-AUC (headline) and average precision / PR-AUC, both robust
  to the 27.0% churn rate. Accuracy and Brier score are
  recorded on the holdout but accuracy alone is not used for the verdict.
- **Test touched once:** the held-out set is scored a single time, after all
  decisions were made.

## Sanity checks (run before believing any result)
- **Baseline floor:** prior-only classifier ROC-AUC =
  0.500 (≈0.5 as expected; majority class
  rate 72.1%). Both models beat it.
- **Label-shuffle:** with shuffled labels, ROC-AUC =
  0.495 (≈0.5 — no signal leaks around the labels).
- **Leakage demonstration:** re-adding `account_status` drives holdout ROC-AUC
  to 1.000 — confirming it is a
  leak and justifying its exclusion.

## Limitations
- The dataset is synthetic; the data-generating process is a logistic function
  of the three retained features, which favors a correctly-specified linear
  model. Results need not transfer to real churn data.
- `signup_date` was treated as noise w.r.t. churn (used only for ordering); a
  real deployment should verify there is no genuine temporal drift in the label.
- No hyperparameter tuning was performed — both models use library defaults
  under equal budget, so this compares default behavior, not tuned ceilings.
- Statistical test is a normal-approximation CI over 5 folds, not a
  large-sample test; treat the CI as indicative.
