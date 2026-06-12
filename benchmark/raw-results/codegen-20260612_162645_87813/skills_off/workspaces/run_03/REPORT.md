# Churn prediction: GradientBoosting vs LogisticRegression

## Claim
Does GradientBoosting outperform LogisticRegression at predicting churn?

## Conclusion
**logistic_regression outperforms gradient_boosting on roc_auc (mean diff -0.0272, p=0.006, n=5)**

On the primary metric (**roc_auc**, threshold-free and robust to the
27% churn base rate), the two models differ by
-0.0272 (gradient_boosting − logistic_regression),
with a paired-fold sd of 0.0117 across n=5 folds
(paired t-test p=0.006). The per-fold difference is small in magnitude but consistent in sign across folds, so the paired t-test resolves it (p=0.006): **logistic_regression** is the better model on this dataset by 0.0272 AUC. The effect is modest — read it as a reliable but small edge, not a large one.

## Results

| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | accuracy (mean ± sd) |
|---|---|---|---|
| logistic_regression | 0.7329 ± 0.0252 | 0.5014 ± 0.0415 | 0.7489 ± 0.0230 |
| gradient_boosting | 0.7057 ± 0.0190 | 0.4704 ± 0.0238 | 0.7318 ± 0.0290 |

Majority-class baseline accuracy ≈ 0.7306; report AUC, not accuracy, as
the primary metric because accuracy can look strong simply by predicting "no churn".

## Methodology

**Question framed as an argument.** The single variable is the model type; the
feature set, split, preprocessing, and seeds are held fixed across both arms.

**Data discipline (applied before any model sees the data):**
- Raw rows: 4200; after removing 200 exact
  duplicate rows: 4000. Deduplication happens *before* splitting
  so identical rows cannot straddle train/test and inflate scores.
- Dropped columns (with reasons):
  - `account_status`: deterministic function of the label (closed iff churned)
  - `customer_id`: row identifier, no predictive signal
  `account_status` is a perfect target leak (it equals "closed" iff the customer
  churned), so including it would make the task trivially solvable and prove nothing.
- Features used: tenure_months, monthly_spend, support_tickets.

**Split — time-aware.** churn is forward-looking, so rows are ordered by
`signup_date` and evaluated with `TimeSeriesSplit` (5 folds):
every fold trains on earlier customers and tests on later ones. A random split on
this temporal data would be leakage. The folds double as repetition, giving
n=5 measurements per arm.

**Preprocessing fit on train only.** `StandardScaler` lives inside the pipeline,
so it is fit on each training fold and applied to the held-out fold
(split-before-transform).

**Reproducibility.** seed=42; data generated with
`python3 make_dataset.py --out churn.csv --seed 7`; code revision `e11943a`. Re-running with
the same seed produces identical metrics.

## Sanity checks (run before believing the comparison)
- **dedup**: PASS — 200 exact duplicates removed at load; 0 remain
- **baseline_floor**: PASS — all arms above AUC 0.5 baseline: logistic_regression=0.733, gradient_boosting=0.706
- **leakage_ceiling**: PASS — all arms below AUC 0.95 (would imply leakage): logistic_regression=0.733, gradient_boosting=0.706
- **label_shuffle**: PASS — shuffled-label AUC near 0.5: logistic_regression=0.476, gradient_boosting=0.490
- **overfit_tiny_subset**: PASS — tiny-subset train AUC near 1.0: logistic_regression=0.903, gradient_boosting=1.000

## Limitations
- **Small n for the statistical test.** The comparison rests on
  5 time-series folds. The paired t-test is valid but
  low-powered: a *significant* result here reflects a consistent sign across folds
  rather than a large effect, and a *null* result would mean "not resolved at this
  budget", not "provably equal". Treat any winner as a small, dataset-specific edge.
- **No hyperparameter tuning.** Both models use fixed, reasonable defaults under a
  shared CPU budget. A tuned GradientBoosting (or a regularized LR) could shift the
  result; any such tuning must be done on validation folds, never on the held-out
  fold, to keep the comparison honest.
- **Synthetic data.** The generator's churn signal is (by construction) a logistic
  function of the features, which structurally favors a linear model; real churn
  data with interactions could change the ranking.
- **Single data seed.** Results are reported for one generated dataset; a different
  `--data-seed` would give a fresh draw.
