# Churn prediction: Gradient Boosting vs Logistic Regression

## Claim under test

For predicting `churned` on this dataset, does `GradientBoostingClassifier` outperform `LogisticRegression` on out-of-sample ROC AUC?

## Conclusion

**Logistic regression outperforms gradient boosting by 0.0181 AUC (+/-0.0101) across 5 folds.**

- Logistic regression: ROC AUC **0.7329 ± 0.0252** (n=5 time folds)
- Gradient boosting:   ROC AUC **0.7148 ± 0.0221** (n=5 time folds)
- Paired per-fold gap (GBM − LogReg): -0.0181 ± 0.0101

A winner is claimed only when the per-fold gap's ±1 sd band excludes zero. Otherwise the honest statement is *no detectable difference*.

## Methodology

- **Evaluation:** TimeSeriesSplit (forward-chaining on signup_date order), 5 folds. The dataset carries a temporal column (`signup_date`) and churn is forward-looking, so a random split would leak future information. Rows are ordered by signup date and each fold trains on the past, validates on the future.
- **No hyperparameter tuning.** Both models use fixed library defaults (LogReg `max_iter=1000`). Because nothing is selected on the validation folds, every fold score is legitimately out-of-sample and the CV mean is an unbiased estimate — no separate held-out test is consumed by tuning.
- **Features:** `tenure_months`, `monthly_spend`, `support_tickets`. Scaling (`StandardScaler`) is fit on the training fold only, inside a `Pipeline`, so no validation statistics leak into fitting.
- **Primary metric:** ROC AUC (threshold-free; survives the 27.1% positive rate). Average precision and accuracy are reported for context but accuracy alone is not trusted under imbalance.
- **Seeds:** all randomness pinned to seed 7 (GBM `random_state`; LogReg is deterministic). Re-runs are identical.

## Data handling and leak surface

The following decisions were made **before** modeling, by inspecting the data, and are defended by the sanity checks below:

- **`account_status` dropped (target leak).** It equals `"closed"` iff the customer churned — a perfect proxy for the label. Including it drives AUC to **1.0000** (leakage-ceiling check), which is why it is excluded from all real arms.
- **200 exact duplicate rows removed before splitting** (4200 → 4000 rows). Dedup precedes the split so duplicates cannot straddle train/validation.
- **`signup_date` used only for time ordering**, never as a feature. **`customer_id` dropped** as a non-predictive identifier.
- **Class balance:** churn rate is 27.1% (a fact, not a footnote) — metrics chosen accordingly.

## Sanity checks (run before trusting the comparison)

- **Baseline floor** (DummyClassifier, prior): AUC 0.5000 ≈ 0.5. Both models clear it.
- **Label-shuffle test:** GBM on shuffled labels gives AUC 0.4878 ≈ 0.5 — no information leaks around the labels.
- **Leakage ceiling:** adding `account_status` back gives AUC 1.0000 ≈ 1.0 — confirms it is a leak and validates the decision to drop it.

## Per-fold ROC AUC

- Logistic regression: 0.7344, 0.7372, 0.6951, 0.7659, 0.7318
- Gradient boosting:   0.7032, 0.7137, 0.6876, 0.7465, 0.7230

## Limitations

- The underlying signal is a linear function of the features (by construction of the generator), which favors no model family a priori but offers gradient boosting little nonlinearity to exploit. Results may not generalize to datasets with strong feature interactions.
- Only library-default hyperparameters were compared. A tuned GBM (or tuned LogReg) could differ; that would require a nested-CV protocol to avoid touching the evaluation folds during selection.
- 5 time folds is a small sample for variance; the ± sd bands are indicative, not a formal significance test.
- Evaluation is within a single dataset and seed for data generation; no external validation set exists.
