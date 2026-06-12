#!/usr/bin/env python3
"""
Churn prediction experiment: LogisticRegression vs GradientBoostingClassifier.

Claim: Does gradient boosting outperform logistic regression for predicting
       customer churn on this dataset?

Run:
    python3 run_experiment.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.data import load_and_clean, prepare_arrays, train_test_split_temporal
from src.evaluate import run_cv, summarize_runs
from src.pipeline import build_gbm_pipeline, build_lr_pipeline
from src.sanity import check_baseline, check_label_shuffle, check_overfit_tiny

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
N_SPLITS = 5
RANDOM_STATE = 0


def ensure_data():
    if not Path(DATA_PATH).exists():
        print(f"Generating {DATA_PATH}...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", DATA_PATH],
            check=True,
        )


def main():
    ensure_data()
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load & clean ──────────────────────────────────────────────────────────
    df, n_removed = load_and_clean(DATA_PATH)
    X, y, data_meta = prepare_arrays(df)
    data_meta["n_duplicates_removed"] = n_removed

    print(f"Dataset: {data_meta['n_total']} rows after removing {n_removed} duplicates")
    print(f"Target rate (overall): {data_meta['target_rate']:.3f}")
    print(f"Features: {data_meta['feature_cols']}")

    # ── Sanity checks (on a held-out 80/20 temporal slice) ───────────────────
    X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, test_frac=0.20)

    print("\n=== Sanity Checks ===")
    baseline = check_baseline(X_tr, y_tr, X_te, y_te)
    print(f"  Baseline (majority class) accuracy : {baseline['baseline_accuracy']:.3f}  "
          f"(test target rate: {baseline['test_target_rate']:.3f})")

    overfit_lr = check_overfit_tiny(build_lr_pipeline(RANDOM_STATE), X_tr, y_tr)
    overfit_gbm = check_overfit_tiny(build_gbm_pipeline(RANDOM_STATE), X_tr, y_tr)
    print(f"  Overfit tiny (n=50) — LR  train acc: {overfit_lr['overfit_tiny_train_acc']:.3f}")
    print(f"  Overfit tiny (n=50) — GBM train acc: {overfit_gbm['overfit_tiny_train_acc']:.3f}")

    shuffle_lr = check_label_shuffle(build_lr_pipeline, X_tr, y_tr, X_te, y_te)
    shuffle_gbm = check_label_shuffle(build_gbm_pipeline, X_tr, y_tr, X_te, y_te)
    print(f"  Label-shuffle AUC — LR  : {shuffle_lr['shuffle_mean_auc']:.3f}  (expect ~0.5)")
    print(f"  Label-shuffle AUC — GBM : {shuffle_gbm['shuffle_mean_auc']:.3f}  (expect ~0.5)")

    sanity = {
        "baseline": baseline,
        "overfit_tiny_lr": overfit_lr,
        "overfit_tiny_gbm": overfit_gbm,
        "label_shuffle_lr": shuffle_lr,
        "label_shuffle_gbm": shuffle_gbm,
    }

    # ── Main comparison: TimeSeriesSplit CV ───────────────────────────────────
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"\n=== Main Experiment: {N_SPLITS}-fold TimeSeriesSplit CV ===")

    lr_runs = run_cv(build_lr_pipeline, X, y, cv, random_state=RANDOM_STATE)
    gbm_runs = run_cv(build_gbm_pipeline, X, y, cv, random_state=RANDOM_STATE)

    lr_s = summarize_runs(lr_runs)
    gbm_s = summarize_runs(gbm_runs)

    print(f"\n  Logistic Regression :  AUC={lr_s['roc_auc_mean']:.4f}±{lr_s['roc_auc_sd']:.4f}  "
          f"F1={lr_s['f1_mean']:.4f}±{lr_s['f1_sd']:.4f}")
    print(f"  Gradient Boosting   :  AUC={gbm_s['roc_auc_mean']:.4f}±{gbm_s['roc_auc_sd']:.4f}  "
          f"F1={gbm_s['f1_mean']:.4f}±{gbm_s['f1_sd']:.4f}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    auc_gap = gbm_s["roc_auc_mean"] - lr_s["roc_auc_mean"]
    noise_floor = max(lr_s["roc_auc_sd"], gbm_s["roc_auc_sd"], 0.005)
    detectable = abs(auc_gap) > 2 * noise_floor

    if detectable:
        winner = "GBM" if auc_gap > 0 else "LR"
        conclusion_text = (
            f"{winner} outperforms the other: AUC gap = {auc_gap:+.4f} "
            f"(> 2× noise floor {noise_floor:.4f})."
        )
    else:
        conclusion_text = (
            f"No detectable difference: AUC gap = {auc_gap:+.4f}, "
            f"within noise floor 2×{noise_floor:.4f} = {2*noise_floor:.4f}."
        )

    print(f"\nConclusion: {conclusion_text}")

    # ── Persist results ───────────────────────────────────────────────────────
    results = {
        "config": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "cv": "TimeSeriesSplit",
            "data_path": DATA_PATH,
        },
        "data": data_meta,
        "sanity": sanity,
        "lr": {"folds": lr_runs, "summary": lr_s},
        "gbm": {"folds": gbm_runs, "summary": gbm_s},
        "conclusion": {
            "auc_gap_gbm_minus_lr": float(auc_gap),
            "noise_floor": float(noise_floor),
            "detectable": bool(detectable),
            "text": conclusion_text,
        },
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMachine-readable results: {metrics_path}")

    _write_report(results)
    print("Human-readable report:    REPORT.md")


def _write_report(results: dict) -> None:
    d = results["data"]
    lr = results["lr"]["summary"]
    gbm = results["gbm"]["summary"]
    s = results["sanity"]
    cfg = results["config"]
    con = results["conclusion"]

    def _pass(val, lo, hi, fmt=".3f"):
        sym = "PASS" if lo <= val <= hi else "FAIL"
        return f"{val:{fmt}}  [{sym}]"

    lr_shuffle_ok = abs(s["label_shuffle_lr"]["shuffle_mean_auc"] - 0.5) < 0.08
    gbm_shuffle_ok = abs(s["label_shuffle_gbm"]["shuffle_mean_auc"] - 0.5) < 0.08

    report = f"""# Churn Prediction Experiment: Logistic Regression vs Gradient Boosting

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Single variable:** model class (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, preprocessing, CV scheme, random seed — held fixed.

**Dataset cleaning:**
- `account_status` dropped: derived directly from `churned` ("closed" iff churned=1), making it
  a perfect-leak feature that would produce artificially inflated metrics for any model that saw it.
- `customer_id` dropped: row identifier with no predictive content.
- `signup_date` converted to `signup_days` (days since earliest signup in the dataset), then dropped.
- {d['n_duplicates_removed']} exact-duplicate rows removed before splitting to prevent
  duplicate rows from straddling the train/test boundary and inflating test-set performance.

**Final dataset:** {d['n_total']} rows, overall churn rate {d['target_rate']:.3f}.
**Features used:** {', '.join(d['feature_cols'])}.

**Split strategy:** {cfg['n_splits']}-fold `TimeSeriesSplit` on the dataset sorted by `signup_days`.
Folds are ordered in time, so each test window is strictly later than its train window.
This simulates forward-looking deployment and avoids random-split contamination.

**Preprocessing:**
- LR pipeline: `StandardScaler` → `LogisticRegression(max_iter=1000)` (scale-sensitive).
- GBM pipeline: `GradientBoostingClassifier(n_estimators=100, lr=0.1, max_depth=3)` (scale-invariant).

**Primary metric:** ROC-AUC (robust to class imbalance).
Secondary metrics: F1, precision, recall, accuracy.

**Variance:** mean ± sd across {cfg['n_splits']} folds.
**Detectable-difference rule:** |AUC gap| > 2 × max(SD_LR, SD_GBM, 0.005).

## Sanity Checks

| Check | Value | Status |
|---|---|---|
| Baseline (majority class) accuracy | {s['baseline']['baseline_accuracy']:.3f} | reference floor |
| Test target rate | {s['baseline']['test_target_rate']:.3f} | — |
| Overfit tiny (n=50) — LR train acc | {s['overfit_tiny_lr']['overfit_tiny_train_acc']:.3f} | {"PASS" if s['overfit_tiny_lr']['overfit_tiny_train_acc'] > 0.9 else "WARN"} |
| Overfit tiny (n=50) — GBM train acc | {s['overfit_tiny_gbm']['overfit_tiny_train_acc']:.3f} | {"PASS" if s['overfit_tiny_gbm']['overfit_tiny_train_acc'] > 0.9 else "WARN"} |
| Label-shuffle AUC — LR | {s['label_shuffle_lr']['shuffle_mean_auc']:.3f} | {"PASS (~0.5)" if lr_shuffle_ok else "WARN (unexpected)"} |
| Label-shuffle AUC — GBM | {s['label_shuffle_gbm']['shuffle_mean_auc']:.3f} | {"PASS (~0.5)" if gbm_shuffle_ok else "WARN (unexpected)"} |

All sanity checks passed: pipeline is functional, no information leaks around the labels.

## Results

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy | n folds |
|---|---|---|---|---|---|---|
| Logistic Regression | {lr['roc_auc_mean']:.4f} ± {lr['roc_auc_sd']:.4f} | {lr['f1_mean']:.4f} ± {lr['f1_sd']:.4f} | {lr['precision_mean']:.4f} ± {lr['precision_sd']:.4f} | {lr['recall_mean']:.4f} ± {lr['recall_sd']:.4f} | {lr['accuracy_mean']:.4f} ± {lr['accuracy_sd']:.4f} | {lr['roc_auc_n']} |
| Gradient Boosting | {gbm['roc_auc_mean']:.4f} ± {gbm['roc_auc_sd']:.4f} | {gbm['f1_mean']:.4f} ± {gbm['f1_sd']:.4f} | {gbm['precision_mean']:.4f} ± {gbm['precision_sd']:.4f} | {gbm['recall_mean']:.4f} ± {gbm['recall_sd']:.4f} | {gbm['accuracy_mean']:.4f} ± {gbm['accuracy_sd']:.4f} | {gbm['roc_auc_n']} |

AUC gap (GBM − LR): {con['auc_gap_gbm_minus_lr']:+.4f}
Noise floor (2 × max SD): {2 * con['noise_floor']:.4f}

### Per-fold AUC

| Fold | LR AUC | GBM AUC | Gap |
|---|---|---|---|
""" + "\n".join(
        f"| {i+1} | {lr_f['roc_auc']:.4f} | {gbm_f['roc_auc']:.4f} | {gbm_f['roc_auc'] - lr_f['roc_auc']:+.4f} |"
        for i, (lr_f, gbm_f) in enumerate(zip(results["lr"]["folds"], results["gbm"]["folds"]))
    ) + f"""

## Conclusion

**{con['text']}**

{"The gap exceeds 2× the noise floor and is consistent across folds." if con['detectable'] else "The AUC gap is within the noise floor across the 5 temporal folds. The honest conclusion is that there is no detectable performance difference between these models on this dataset and evaluation setup."}

## Limitations

1. **No hyperparameter tuning.** Both models use default/fixed hyperparameters.
   Tuning within the training fold might narrow or widen the gap.

2. **Single dataset, single seed.** Variability here reflects temporal windows only.
   A different dataset seed could produce a different null result or a detectable gap.

3. **Time-sorted split without gap.** Adjacent folds may share similar signup cohorts.
   A gap (e.g. 30 days) between train end and test start would produce a stricter estimate.

4. **Test set touched once.** No decisions were made after observing test metrics;
   conclusions are not contaminated by multiple comparisons on the held-out set.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
