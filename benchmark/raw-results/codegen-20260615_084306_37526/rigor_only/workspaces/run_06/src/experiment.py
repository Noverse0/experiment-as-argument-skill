"""Experiment orchestration: loads data, runs checks, evaluates models."""
import json
import os

from src.evaluate import (
    baseline_evaluate,
    cv_evaluate,
    holdout_evaluate,
    shuffle_label_test,
)
from src.pipeline import build_features, load_and_clean, make_models, temporal_split


def run(data_path: str = "churn.csv") -> dict:
    print("=== Customer Churn: LR vs GradientBoosting Experiment ===\n")

    # 1. Load and deduplicate
    print("[1] Loading data...")
    df, n_dupes = load_and_clean(data_path)
    print(f"    Rows after dedup: {len(df)} ({n_dupes} exact duplicates removed)")

    # 2. Feature engineering
    print("[2] Building features...")
    X, y, dates = build_features(df)
    print(f"    Features used: {list(X.columns)}")
    print(f"    Target rate (churn): {y.mean():.3f}")

    # 3. Temporal 80/20 split by signup_date
    print("[3] Temporal 80/20 split by signup_date...")
    X_train, X_test, y_train, y_test = temporal_split(X, y, dates, train_frac=0.80)
    print(f"    Train: {len(X_train)} rows, churn rate {y_train.mean():.3f}")
    print(f"    Test:  {len(X_test)} rows,  churn rate {y_test.mean():.3f}")

    # 4. Sanity checks
    print("\n[4] Sanity checks...")
    models = make_models()
    baseline = baseline_evaluate(y_train, y_test)
    print(f"    Stratified dummy baseline ROC-AUC: {baseline['roc_auc']:.3f}")

    print("    Running shuffle-label sanity check (LR, 3 seeds × 3 folds)...")
    shuffle = shuffle_label_test(models["LogisticRegression"], X_train, y_train, n_seeds=3)
    mean_auc = shuffle["mean_roc_auc"]
    print(f"    Shuffled-label AUC: {mean_auc:.3f} ± {shuffle['std_roc_auc']:.3f}")

    if mean_auc >= 0.6:
        raise RuntimeError(
            f"Shuffle-label test FAILED: AUC={mean_auc:.3f} >= 0.6. "
            "Features may contain label-independent signal or a leak exists."
        )
    print("    PASS: shuffled-label AUC < 0.6")

    # 5. Cross-validation on training set (3 seeds × 5 folds = 15 evals each)
    print("\n[5] CV evaluation on training set (3 seeds × 5 folds)...")
    cv_results = {}
    for name, pipe in models.items():
        print(f"    {name}...", end=" ", flush=True)
        cv_results[name] = cv_evaluate(pipe, X_train, y_train, n_seeds=3, n_folds=5)
        r = cv_results[name]
        print(
            f"ROC-AUC {r['roc_auc']['mean']:.3f} ± {r['roc_auc']['std']:.3f}  "
            f"AP {r['avg_precision']['mean']:.3f} ± {r['avg_precision']['std']:.3f}"
        )

    # 6. Holdout evaluation (test set touched exactly once)
    print("\n[6] Holdout evaluation on test set (fit on full train, eval once)...")
    holdout_results = {}
    for name, pipe in models.items():
        holdout_results[name] = holdout_evaluate(pipe, X_train, y_train, X_test, y_test)
        r = holdout_results[name]
        print(f"    {name}: ROC-AUC={r['roc_auc']:.3f}, AP={r['avg_precision']:.3f}")

    # 7. Compile and save
    results = {
        "dataset": {
            "path": data_path,
            "n_rows_raw": len(df) + n_dupes,
            "n_rows_after_dedup": len(df),
            "n_dupes_removed": n_dupes,
            "target_rate": float(y.mean()),
            "features_used": list(X.columns),
            "features_dropped": {
                "days_since_last_login": "post-outcome leak",
                "customer_id": "identifier",
                "signup_date": "converted to signup_days numeric feature",
            },
        },
        "split": {
            "method": "temporal_chronological_80_20",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_target_rate": float(y_train.mean()),
            "test_target_rate": float(y_test.mean()),
        },
        "sanity_checks": {
            "baseline_roc_auc": baseline["roc_auc"],
            "baseline_avg_precision": baseline["avg_precision"],
            "shuffle_label_mean_roc_auc": shuffle["mean_roc_auc"],
            "shuffle_label_std_roc_auc": shuffle["std_roc_auc"],
            "shuffle_label_pass": mean_auc < 0.6,
        },
        "cv_evaluation": cv_results,
        "holdout_evaluation": holdout_results,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[7] Saved results/metrics.json")

    _write_report(results)
    print("[8] Saved REPORT.md\n")

    return results


def _write_report(r: dict) -> None:
    ds = r["dataset"]
    sp = r["split"]
    sc = r["sanity_checks"]
    cv = r["cv_evaluation"]
    ho = r["holdout_evaluation"]

    lr_cv = cv["LogisticRegression"]
    gb_cv = cv["GradientBoosting"]
    lr_ho = ho["LogisticRegression"]
    gb_ho = ho["GradientBoosting"]

    gap = gb_ho["roc_auc"] - lr_ho["roc_auc"]
    pooled_std = (lr_cv["roc_auc"]["std"] + gb_cv["roc_auc"]["std"]) / 2

    if abs(gap) < pooled_std:
        conclusion = (
            f"**No detectable difference.** GradientBoosting ROC-AUC gap vs "
            f"LogisticRegression is {gap:+.3f}, within the CV noise "
            f"(pooled std ≈ {pooled_std:.3f}). Neither model is a clear winner "
            f"on this dataset."
        )
    elif gap > 0:
        conclusion = (
            f"**GradientBoosting outperforms LogisticRegression** on the holdout "
            f"set (ROC-AUC gap: {gap:+.3f}). The gap exceeds the CV noise "
            f"(pooled std ≈ {pooled_std:.3f})."
        )
    else:
        conclusion = (
            f"**LogisticRegression matches or outperforms GradientBoosting** "
            f"(ROC-AUC gap: {gap:+.3f}). This is consistent with the underlying "
            f"data-generating process being approximately linear in the honest features."
        )

    report = f"""# Churn Prediction: Logistic Regression vs Gradient Boosting

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion

{conclusion}

## Dataset

| Property | Value |
|----------|-------|
| Source | `{ds['path']}` (generated by `make_dataset.py`) |
| Rows before dedup | {ds['n_rows_raw']} |
| Rows after dedup | {ds['n_rows_after_dedup']} ({ds['n_dupes_removed']} exact duplicates removed) |
| Overall churn rate | {ds['target_rate']:.3f} ({ds['target_rate']*100:.1f}%) |

## Feature Engineering

**Features used ({len(ds['features_used'])}):** {', '.join(f'`{f}`' for f in ds['features_used'])}

**Features dropped:**

| Column | Reason |
|--------|--------|
| `days_since_last_login` | **Post-outcome leak.** A churned customer has stopped logging in, so this value is only knowable *after* the churn event. Including it inflates model performance artificially. |
| `customer_id` | Identifier — no predictive signal. |
| `signup_date` (raw) | Converted to `signup_days` (days from earliest signup) to capture cohort vintage as a numeric feature without carrying a raw date string. |

## Methodology

### Data Preprocessing
1. **Deduplication first:** {ds['n_dupes_removed']} exact duplicate rows removed before any split to prevent duplicates from straddling train/test boundaries.
2. **Temporal split:** Customers are ordered by `signup_date`; the first 80% (by chronological order) form the training set, the last 20% the test set. This simulates predicting churn for newer customer cohorts.
3. **Scaler:** `StandardScaler` fitted on training data only, then applied to both sets. The test distribution never touches the scaler fit.

### Split Summary

| Set | Rows | Churn Rate |
|-----|------|-----------|
| Train | {sp['n_train']} | {sp['train_target_rate']:.3f} |
| Test  | {sp['n_test']}  | {sp['test_target_rate']:.3f}  |

### Models

| Model | Configuration |
|-------|--------------|
| `LogisticRegression` | `max_iter=1000`, `random_state=42` |
| `GradientBoostingClassifier` | `n_estimators=100`, `random_state=42` |

### Evaluation Protocol
- **CV (variance estimate):** 3 seeds × 5-fold StratifiedKFold on the training set → 15 evaluations per model. Reports mean ± std.
- **Holdout:** Each model fit on the full training set, evaluated once on the held-out test set. The test set is touched exactly once.

### Metrics
- **ROC-AUC** (primary): robust to the ~{ds['target_rate']*100:.0f}% class imbalance.
- **Average Precision (AP):** measures performance on the minority (churned) class.

## Sanity Checks

| Check | Result | Status |
|-------|--------|--------|
| Stratified dummy baseline ROC-AUC | {sc['baseline_roc_auc']:.3f} | Baseline established |
| Shuffle-label AUC (should be ≈ 0.5) | {sc['shuffle_label_mean_roc_auc']:.3f} ± {sc['shuffle_label_std_roc_auc']:.3f} | {"PASS" if sc['shuffle_label_pass'] else "FAIL"} |

The shuffle-label test confirms no label-independent leakage in the feature set.

## Results

### Cross-Validation (Training Set, 3 seeds × 5 folds, n=15)

| Model | ROC-AUC mean ± std | AP mean ± std |
|-------|--------------------|--------------|
| LogisticRegression  | {lr_cv['roc_auc']['mean']:.3f} ± {lr_cv['roc_auc']['std']:.3f} | {lr_cv['avg_precision']['mean']:.3f} ± {lr_cv['avg_precision']['std']:.3f} |
| GradientBoosting    | {gb_cv['roc_auc']['mean']:.3f} ± {gb_cv['roc_auc']['std']:.3f} | {gb_cv['avg_precision']['mean']:.3f} ± {gb_cv['avg_precision']['std']:.3f} |

### Holdout Test Set (single evaluation, test touched once)

| Model | ROC-AUC | AP |
|-------|---------|----|
| LogisticRegression  | {lr_ho['roc_auc']:.3f} | {lr_ho['avg_precision']:.3f} |
| GradientBoosting    | {gb_ho['roc_auc']:.3f} | {gb_ho['avg_precision']:.3f} |
| Stratified baseline | {sc['baseline_roc_auc']:.3f} | {sc['baseline_avg_precision']:.3f} |

## Limitations

1. **CV ignores time ordering within the train period.** StratifiedKFold randomly mixes training rows across folds. The holdout result (temporal gap between cohorts) is the more deployment-realistic estimate.
2. **Linear DGP.** The underlying data-generating process is approximately linear (logit = f(tenure, spend, tickets)), which structurally favors logistic regression. On a dataset with non-linear interactions, GB would likely outperform LR.
3. **Default hyperparameters.** Neither model was tuned. Tuning GB (tree depth, learning rate) could improve its performance but would require a separate validation set.
4. **Single holdout run.** The holdout point estimate has no confidence interval. The CV std partially addresses this within the training period only.
5. **Dataset size (~{ds['n_rows_after_dedup']} rows).** Small datasets limit GB's capacity advantage over parametric models.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)
