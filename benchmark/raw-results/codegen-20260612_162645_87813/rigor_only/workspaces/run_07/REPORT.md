# Churn: Gradient Boosting vs Logistic Regression

## Claim under test

Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting `churned` on this dataset?

## Conclusion

**Logistic regression is better** on ROC-AUC: mean paired delta -0.0179 (sd 0.0102, n=5, paired t-test p=0.017). LR=0.7328±0.0252, GB=0.7148±0.0221.

## Methodology

- **Single variable:** the classifier. Both arms share identical folds, features, preprocessing, and seed (7).
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Columns dropped and why:**
  - `account_status` — target leak (closed iff churned)
  - `customer_id` — row identifier, no signal
  - `signup_date` — temporal ordering only, not a feature
- **Duplicates:** 200 exact duplicate rows removed before splitting (raw 4200 → clean 4000); leaving them would let identical rows straddle train/test.
- **Split:** TimeSeriesSplit on signup_date order (train=past, test=future), 5 folds. Churn is forward-looking, so a random split would train on the future — we split by time instead.
- **Preprocessing:** StandardScaler fit on train fold only (in Pipeline) — no fit-like step ever sees test rows.
- **Metrics:** ROC-AUC (primary) and PR-AUC, both threshold-free and robust to the 27% positive rate. Accuracy alone would be misleading.
- **Variance:** 5 paired folds per arm; we report mean ± sd and a paired t-test on per-fold ROC-AUC, not a single number.

## Sanity checks (run before trusting the comparison)

- **Majority baseline** ROC-AUC = 0.500 (expected ≈ 0.5) — models must beat this.
- **Label shuffle** ROC-AUC = 0.493 (expected ≈ 0.5) — confirms no information leaks around the labels.
- **Overfit tiny slice** train ROC-AUC = 1.000 (expected high) — the pipeline can actually learn.
- **Leakage ceiling** (with `account_status` added back) ROC-AUC = 1.000 (≈ 1.0) — direct evidence that `account_status` is a target leak, justifying its removal.

## Results

| Arm | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | n folds |
|---|---|---|---|
| Logistic regression | 0.7328 ± 0.0252 | 0.5014 ± 0.0412 | 5 |
| Gradient boosting | 0.7148 ± 0.0221 | 0.4782 ± 0.0302 | 5 |

Paired ROC-AUC delta (GB − LR): -0.0179 ± 0.0102, p = 0.017.

## Limitations

- **One dataset, one generation seed.** Variance here is across time folds, not across resampled datasets; the estimate of generalization is correspondingly narrow.
- **Small fold count (n=5).** The paired test has low power; a true small effect could be missed (Type II), so 'no detectable difference' means *not detectable at this n*, not 'provably equal'.
- **Time-based folds vary in size.** Early folds train on fewer rows; AUC sd partly reflects that, not only model variability.
- **Default GB hyperparameters.** Neither model was tuned; tuning budget was held at zero for both to keep the comparison fair. Results may shift under tuning.
- **Synthetic, logistic-generated labels.** The data-generating process is linear in log-odds, which structurally favours logistic regression; a real churn signal with interactions could change the verdict.

_Provenance: data sha256 68a1ead7edfa3c04, code e11943a, sklearn 1.7.1, python 3.12.4._
