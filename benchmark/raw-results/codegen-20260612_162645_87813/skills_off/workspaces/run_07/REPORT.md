# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does GradientBoostingClassifier outperform LogisticRegression at predicting churn?

## Conclusion
**logistic_regression is better on ROC-AUC: mean(gb - lr) = -0.0223 (sd 0.0112, n=5 folds, paired t p=0.011).**

PR-AUC: mean(gb - lr) = -0.0265 (sd 0.0198, p=0.040). F1: mean(gb - lr) = +0.0320 (sd 0.0324, p=0.092).

## Results (mean ± sd across 5 time folds)
| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| logistic_regression | 0.7329 ± 0.0252 | 0.5014 ± 0.0415 | 0.3694 ± 0.0401 |
| gradient_boosting | 0.7106 ± 0.0198 | 0.4749 ± 0.0271 | 0.4014 ± 0.0335 |

Baseline (prior-only dummy) ROC-AUC ≈ 0.50 by construction — both models clear it.

## Methodology
- **Single variable:** the classifier. Both arms share identical preprocessing
  (`StandardScaler` → classifier in a `Pipeline`) and the same folds, so any
  difference is attributable to the model, not the pipeline.
- **Leakage controls (measured, not assumed):**
  - Dropped `account_status` — a *perfect* target leak: leak fraction
    1.000
    (= "closed" iff churned). Kept, it drives ROC-AUC to 1.0 and proves nothing.
  - Dropped `customer_id` (identifier).
  - Removed 200 exact duplicate rows *before* splitting
    (4200 → 4000 rows) so no row straddles train/test.
  - `signup_date` is temporal and churn is forward-looking, so we sort by it and
    use **TimeSeriesSplit** (forward-chaining): every test fold lies strictly
    after its training window. A random split would leak the future.
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Metrics:** ROC-AUC (primary; threshold-free, robust to the
  27.1% churn imbalance), PR-AUC, and F1 at threshold 0.5.
- **Repetition & comparison:** 5 folds per arm,
  paired by fold; reported as mean ± sd with a paired t-test. An interval that
  crosses zero is called "no detectable difference" — no winner claim without
  variance.
- **Seeds:** global seed 42 (model `random_state`);
  label-shuffle seed 123. Re-running with the same seed reproduces the numbers.

## Sanity checks (run before the comparison)
- **leak_excluded**: PASS (account_status_leak_fraction=1.000, account_status_in_features=0.000)
- **baseline_floor**: PASS (baseline_roc_auc=0.500)
- **label_shuffle**: PASS (shuffled_roc_auc=0.472)
- **overfit_tiny**: PASS (train_roc_auc=1.000, n=60.000)

All passed: **True**. The label-shuffle collapsing to
~0.5 and the prior baseline sitting at ~0.5 together argue the reported signal
is real and not residual leakage; the tiny-slice overfit confirms the pipeline
can actually learn.

## Limitations
- **Weak, near-linear signal.** The target is generated from a logistic function
  of the three features, so there is little non-linear structure for boosting to
  exploit — this dataset is close to a best case for logistic regression. The
  null/near-null result should not be generalized to richer real-world churn data.
- **Few folds (n=5).** The paired t-test has low
  power; "no detectable difference" means *not detectable at this sample size*,
  not "provably equal".
- **No hyperparameter search.** Both models use fixed, reasonable defaults under
  an equal (zero) tuning budget. A tuned GBM might separate from LR; that is a
  different experiment and would require a held-out tuning split.
- **Test contact:** the final fold metrics were read once to write this report;
  no model or feature decision was made after seeing them.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py --data churn.csv
```
Environment: Python 3.12.4, scikit-learn 1.7.1.
