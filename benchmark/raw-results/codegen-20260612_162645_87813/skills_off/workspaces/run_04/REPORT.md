# Churn prediction: gradient boosting vs logistic regression

## Claim
Gradient boosting outperforms logistic regression for churn prediction.

## Verdict
**logistic regression wins on this run: ROC-AUC gap (gboost - logreg) -0.0177 +/- 0.0093 (n=15), spread does not cross zero.**

## Result
Repetition unit: one (seed, forward-fold) pair. 3 seeds x 5 folds = 15 paired measurements per arm.

| arm | ROC-AUC (mean +/- sd) | Avg precision (mean +/- sd) | n |
|-----|----------------------|------------------------------|---|
| logreg | 0.7327 +/- 0.0234 | 0.5014 +/- 0.0380 | 15 |
| gboost | 0.7150 +/- 0.0205 | 0.4783 +/- 0.0276 | 15 |

Paired ROC-AUC difference (gboost - logreg): **-0.0177 +/- 0.0093** (n=15).
Both arms clear the no-skill baseline (ROC-AUC 0.500); the target rate is 0.271, so average precision is reported alongside ROC-AUC because accuracy alone would be misleading under this imbalance.

## Methodology
- **Variable:** estimator (LogisticRegression vs GradientBoostingClassifier). Held fixed: preprocessing (StandardScaler), features, folds, tuning budget (library defaults).
- **Features used:** tenure_months, monthly_spend, support_tickets. `customer_id` (identifier) and `signup_date` (used only as the time axis) are excluded as features.
- **Data-contact policy & cleaning** (applied before any split):
  - Dropped target-leak column(s) ['account_status']: in this dataset `account_status == "closed"` iff the customer churned, i.e. it is recorded *after* the outcome. The leakage-ceiling check below quantifies the fake signal it carries.
  - Dropped 200 exact duplicate rows before splitting so no observation straddles the train/test boundary (4200 raw -> 4000 rows).
  - `signup_date` is temporal and the task is forward-looking, so the split is **TimeSeriesSplit(n_splits=5) on signup_date-sorted rows (forward-looking)** rather than random: every fold trains on the past and is scored on a strictly later block. Preprocessing (`StandardScaler`) is fit on the training fold only, inside the pipeline.
- **Comparison:** both arms are scored on identical folds (paired), across seeds [0, 1, 2], so any gap is attributable to the estimator, not to luckier splits.
- **Metrics:** roc_auc, average_precision (threshold-free; survive the ~27% positive rate).

## Sanity checks (run before believing the comparison)
| check | value | expectation |
|-------|-------|-------------|
| baseline floor (no-skill) | ROC-AUC 0.500 | ~0.50 |
| leakage ceiling (account_status alone) | ROC-AUC 1.000 | ~1.00 -> confirms the drop |
| overfit tiny subset (60 rows) | train ROC-AUC 1.000 | ~1.00 |
| label shuffle | ROC-AUC 0.512 | ~0.50 |

All checks passed, so the pipeline is not silently leaking and the model has capacity.

## Limitations
- The honest verdict is read off the paired-difference spread (mean +/- 1 sd straddling zero => "no detectable difference"). With n=15 this is a descriptive spread, not a formal significance test; a larger study would add a paired test and a confidence interval.
- Library-default hyperparameters for both arms (equal, untuned tuning budget). Tuning could move either arm; doing so fairly would require a nested validation split and is out of scope here.
- The data are synthetic and generated from a *logistic* relationship (see `make_dataset.py`), which a priori favours the linear model; the conclusion is about this dataset, not churn in general.
- Forward-looking folds mean early folds train on little data; per-fold variance partly reflects fold size, not only the estimator.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```
Seeds [0, 1, 2], scikit-learn 1.7.1, Python 3.12.4. Re-running with the same seeds reproduces the metrics exactly.
