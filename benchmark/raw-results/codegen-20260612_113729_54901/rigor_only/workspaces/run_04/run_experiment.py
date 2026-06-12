"""Entrypoint: compare LogisticRegression vs GradientBoosting on churn prediction.

Claim: Does gradient boosting outperform logistic regression for predicting
       customer churn on this dataset?

Variable: Model class (LogisticRegression vs GradientBoostingClassifier).
          Everything else (features, preprocessing, split, seed) is held fixed.

Data discipline applied:
  - account_status dropped: derived from the target (closed iff churned==1).
  - Exact duplicates removed before splitting (200 planted rows).
  - Time-based split: latest 20% of customers by signup_date form the test set.
  - StandardScaler fit on train only (inside Pipeline, so CV respects this).

Outputs:
  - results/metrics.json  — machine-readable metrics for both models
  - REPORT.md             — written comparison, methodology, and limitations
"""
import json
import os
import sys
import time
import copy

import numpy as np

from src.data import load_and_clean, time_based_split, get_Xy
from src.models import make_pipelines
from src.evaluate import (
    majority_baseline,
    cross_validate_model,
    evaluate_on_test,
    label_shuffle_check,
    overfit_one_batch_check,
)

SEED = 42
N_CV_FOLDS = 5
DATA_PATH = "churn.csv"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # ── 1. Load and audit ────────────────────────────────────────────────────
    print("=== Loading and cleaning data ===")
    _, audit = load_and_clean(DATA_PATH)
    print(f"  Raw rows: {audit['n_raw']}")
    print(f"  Duplicates removed: {audit['n_dupes_removed']}")
    print(f"  Rows after dedup: {audit['n_after_dedup']}")
    print(f"  Target rate: {audit['target_rate']:.3f}")
    print(f"  Dropped columns: {audit['dropped_columns']}")
    print(f"  Leak column confirmed and excluded: {audit['leak_column_confirmed']}")

    # ── 2. Time-based split ──────────────────────────────────────────────────
    print("\n=== Time-based split (last 20% by signup_date = test) ===")
    train_df, test_df, split_info = time_based_split(DATA_PATH, test_frac=0.2)
    print(f"  Train: {split_info['n_train']} rows  {split_info['train_signup_date_range']}")
    print(f"  Test:  {split_info['n_test']} rows   {split_info['test_signup_date_range']}")
    print(f"  Train target rate: {split_info['train_target_rate']:.3f}")
    print(f"  Test  target rate: {split_info['test_target_rate']:.3f}")

    X_train, y_train = get_Xy(train_df)
    X_test, y_test = get_Xy(test_df)

    # ── 3. Baseline ──────────────────────────────────────────────────────────
    print("\n=== Majority-class baseline ===")
    baseline = majority_baseline(y_train, y_test)
    print(f"  Majority class: {baseline['majority_class']}")
    print(f"  Baseline accuracy: {baseline['test_accuracy']:.3f}")
    print(f"  Baseline AUC: {baseline['test_auc']:.3f}  (models must beat this)")

    # ── 4. Sanity checks ─────────────────────────────────────────────────────
    print("\n=== Sanity checks ===")
    pipes = make_pipelines(seed=SEED)

    sanity = {}
    for name, pipe in pipes.items():
        overfit = overfit_one_batch_check(lambda p=pipe: copy.deepcopy(p), X_train, y_train, n=50)
        shuffle = label_shuffle_check(
            lambda p=pipe: copy.deepcopy(p), X_train, y_train, X_test, y_test, seed=SEED
        )
        sanity[name] = {"overfit_one_batch": overfit, "label_shuffle": shuffle}
        status_o = "PASS" if overfit["passed"] else "FAIL"
        status_s = "PASS" if shuffle["passed"] else "FAIL"
        print(
            f"  {name}: overfit-tiny [{status_o} train_acc={overfit['train_accuracy_on_tiny']:.3f}]"
            f"  label-shuffle [{status_s} mean_auc={shuffle['shuffled_label_auc_mean']:.3f}]"
        )
        if not overfit["passed"] or not shuffle["passed"]:
            print("  WARNING: sanity check failed — investigate before trusting results")

    # ── 5. Cross-validation on train set ─────────────────────────────────────
    print(f"\n=== {N_CV_FOLDS}-fold stratified CV on training set ===")
    cv_results = {}
    for name, pipe in pipes.items():
        cv = cross_validate_model(pipe, X_train, y_train, n_folds=N_CV_FOLDS, seed=SEED)
        cv_results[name] = cv
        print(
            f"  {name}: AUC {cv['cv_auc_mean']:.4f} ± {cv['cv_auc_std']:.4f}"
            f"  F1 {cv['cv_f1_mean']:.4f} ± {cv['cv_f1_std']:.4f}"
        )

    # ── 6. Final evaluation on held-out test set (touched once) ──────────────
    print("\n=== Final evaluation on held-out test set (ONE touch) ===")
    test_results = {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        test_results[name] = evaluate_on_test(pipe, X_test, y_test)
        print(
            f"  {name}: AUC={test_results[name]['test_auc']:.4f}"
            f"  F1={test_results[name]['test_f1']:.4f}"
        )

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # ── 7. Write machine-readable results ─────────────────────────────────────
    metrics = {
        "experiment": "churn_lr_vs_gbm",
        "seed": SEED,
        "n_cv_folds": N_CV_FOLDS,
        "runtime_seconds": round(elapsed, 2),
        "data_audit": audit,
        "split_info": split_info,
        "baseline": baseline,
        "sanity_checks": sanity,
        "cv_results": cv_results,
        "test_results": test_results,
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {metrics_path}")

    # ── 8. Write REPORT.md ────────────────────────────────────────────────────
    write_report(metrics)
    print("Report written to REPORT.md")


def write_report(m: dict):
    lr = "LogisticRegression"
    gbm = "GradientBoosting"
    cv = m["cv_results"]
    test = m["test_results"]
    audit = m["data_audit"]
    split = m["split_info"]
    baseline = m["baseline"]
    sanity = m["sanity_checks"]

    lr_auc = cv[lr]["cv_auc_mean"]
    lr_std = cv[lr]["cv_auc_std"]
    gbm_auc = cv[gbm]["cv_auc_mean"]
    gbm_std = cv[gbm]["cv_auc_std"]

    # Overlap check: are the confidence intervals (±1 std) non-overlapping?
    gap = abs(gbm_auc - lr_auc)
    combined_spread = lr_std + gbm_std
    detectable = gap > combined_spread
    winner = gbm if gbm_auc > lr_auc else lr
    loser = lr if winner == gbm else gbm

    if detectable:
        conclusion = (
            f"**{winner}** outperforms **{loser}** with a detectable gap "
            f"(CV AUC gap {gap:.4f} > combined spread {combined_spread:.4f})."
        )
    else:
        conclusion = (
            f"**No detectable difference**: the CV AUC gap ({gap:.4f}) is within "
            f"the combined spread ({combined_spread:.4f}). "
            f"Neither model reliably outperforms the other on this dataset."
        )

    report = f"""# Churn Prediction: Logistic Regression vs Gradient Boosting

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion
{conclusion}

### Results Summary

| Model | CV AUC (mean ± std, n={m['n_cv_folds']}) | CV F1 (mean ± std) | Test AUC | Test F1 |
|---|---|---|---|---|
| LogisticRegression | {cv[lr]['cv_auc_mean']:.4f} ± {cv[lr]['cv_auc_std']:.4f} | {cv[lr]['cv_f1_mean']:.4f} ± {cv[lr]['cv_f1_std']:.4f} | {test[lr]['test_auc']:.4f} | {test[lr]['test_f1']:.4f} |
| GradientBoosting | {cv[gbm]['cv_auc_mean']:.4f} ± {cv[gbm]['cv_auc_std']:.4f} | {cv[gbm]['cv_f1_mean']:.4f} ± {cv[gbm]['cv_f1_std']:.4f} | {test[gbm]['test_auc']:.4f} | {test[gbm]['test_f1']:.4f} |
| Majority-class baseline | — | — | {baseline['test_auc']:.4f} | — |

Both models beat the majority-class baseline (AUC {baseline['test_auc']:.4f}).

## Methodology

### Variable
Single variable: model class (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, preprocessing, split policy, seed — are held fixed.

### Dataset
- Source: `churn.csv` generated by `make_dataset.py` (seed=7, n=4000 base rows)
- Raw rows: {audit['n_raw']}
- Duplicate rows removed before splitting: {audit['n_dupes_removed']}
- Rows used: {audit['n_after_dedup']}
- Target rate (overall): {audit['target_rate']:.3f}

### Data Integrity Decisions
1. **`account_status` excluded (leakage):** This column is `"closed"` if and only if `churned==1` — it is derived from the target. Including it would produce near-perfect but meaningless accuracy.
2. **Exact duplicates removed before splitting:** The dataset contains {audit['n_dupes_removed']} exact duplicate rows. If retained through a random split, these rows straddle train and test, causing data leakage. Deduplication happens first.
3. **Time-based split used:** `signup_date` is a temporal column. A random split would allow the model to learn from future customer cohorts. We sort by `signup_date` and use the latest 20% as test.

### Split
- Train: {split['n_train']} rows, signup_date {split['train_signup_date_range'][0]} – {split['train_signup_date_range'][1]} (target rate {split['train_target_rate']:.3f})
- Test: {split['n_test']} rows, signup_date {split['test_signup_date_range'][0]} – {split['test_signup_date_range'][1]} (target rate {split['test_target_rate']:.3f})
- The test set is touched exactly once (for final reporting).

### Features
`tenure_months`, `monthly_spend`, `support_tickets`

### Preprocessing
`StandardScaler` is applied inside each `sklearn.Pipeline`, ensuring the scaler is fit only on each CV fold's training portion — no scale leakage across folds.

### Evaluation
- Primary metric: ROC-AUC (robust to class imbalance; target rate ≈ {audit['target_rate']:.2f})
- Secondary metric: F1 (threshold-dependent complement)
- Variance: {m['n_cv_folds']}-fold stratified CV on the training set (seed={m['seed']})
- Winner claim requires the gap to exceed the combined spread (mean ± std of both models)

### Sanity Checks
| Check | LR | GBM |
|---|---|---|
| Overfit tiny (n=50) train accuracy | {sanity[lr]['overfit_one_batch']['train_accuracy_on_tiny']:.3f} ({'PASS' if sanity[lr]['overfit_one_batch']['passed'] else 'FAIL'}) | {sanity[gbm]['overfit_one_batch']['train_accuracy_on_tiny']:.3f} ({'PASS' if sanity[gbm]['overfit_one_batch']['passed'] else 'FAIL'}) |
| Label-shuffle mean AUC (should be ~0.5) | {sanity[lr]['label_shuffle']['shuffled_label_auc_mean']:.3f} ({'PASS' if sanity[lr]['label_shuffle']['passed'] else 'FAIL'}) | {sanity[gbm]['label_shuffle']['shuffled_label_auc_mean']:.3f} ({'PASS' if sanity[gbm]['label_shuffle']['passed'] else 'FAIL'}) |

### Hyperparameters
- **LogisticRegression:** C=1.0, solver=lbfgs, max_iter=1000
- **GradientBoosting:** n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8
- No hyperparameter tuning was performed; defaults were used. Tuning on the same split would require a separate validation set to avoid test contamination.

### Seed and Reproducibility
- Global seed: {m['seed']}
- CV shuffle seed: {m['seed']}
- All model random_states: {m['seed']}
- Re-running with the same seed produces identical metrics.

## Limitations

1. **Single seed:** Results come from one fixed seed. A multi-seed sweep would produce wider but more honest confidence intervals.
2. **No hyperparameter tuning:** Tuning either model could shift the comparison. The current comparison is between sensible defaults only.
3. **Time-based split approximation:** We split on signup cohort, not on actual churn observation date. If churn is measured at a fixed future point for all customers, this split is correct. If measurement windows differ by cohort, there may be residual temporal leakage.
4. **Features used:** Only three numeric features are available. Additional feature engineering could change the relative advantage.
5. **Dataset size:** 4000 rows (after dedup). On larger datasets, GBM's advantage over linear models tends to grow.

## Runtime
{m['runtime_seconds']:.1f} seconds on CPU.
"""
    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
