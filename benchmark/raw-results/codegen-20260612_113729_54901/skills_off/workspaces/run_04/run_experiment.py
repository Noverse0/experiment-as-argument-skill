#!/usr/bin/env python3
"""
Full experiment entrypoint.

Usage:
    python3 run_experiment.py [--data churn.csv]

Writes:
    results/metrics.json   - machine-readable per-model metrics
    REPORT.md              - human-readable comparison and conclusion
"""
import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np

from src.data import load_and_clean, time_split, get_X_y, FEATURE_COLS, TARGET_COL
from src.models import MODELS
from src.evaluate import cv_score, final_test_score, baseline_score, label_shuffle_auc, SEEDS, CV_FOLDS


def fmt_metric(m_dict):
    return f"{m_dict['mean']:.4f} ± {m_dict['std']:.4f} (n={m_dict['n']})"


def main(data_path: str = "churn.csv"):
    print("=== Churn Prediction Experiment ===")
    print(f"Data: {data_path}")

    # --- 1. Load and clean ---
    df, dedup_removed = load_and_clean(data_path)
    print(f"Rows after dedup: {len(df)} ({dedup_removed} duplicates removed)")

    target_rate = df[TARGET_COL].mean()
    print(f"Target rate (churn): {target_rate:.3f}")

    # --- 2. Time-based split ---
    train_df, test_df = time_split(df, test_frac=0.2)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train date range: {train_df['signup_date'].min().date()} – {train_df['signup_date'].max().date()}")
    print(f"Test  date range: {test_df['signup_date'].min().date()} – {test_df['signup_date'].max().date()}")
    print(f"Train churn rate: {train_df[TARGET_COL].mean():.3f} | Test churn rate: {test_df[TARGET_COL].mean():.3f}")

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    # --- 3. Sanity checks ---
    print("\n--- Sanity checks ---")

    # Baseline floor
    base = baseline_score(X_train, y_train, X_test, y_test)
    print(f"Majority-class baseline AUC: {base['roc_auc']:.4f}")

    # Label shuffle test (using LR as representative model)
    shuffle_auc = label_shuffle_auc(MODELS["LogisticRegression"], X_train, y_train, X_test, y_test)
    print(f"Label-shuffle AUC (should be ~0.5): {shuffle_auc:.4f}")
    if shuffle_auc > 0.6:
        print("WARNING: label-shuffle AUC > 0.6 — suspect leakage in features!")

    # --- 4. Cross-validated evaluation (3 seeds × 5-fold CV) ---
    print(f"\n--- CV evaluation ({len(SEEDS)} seeds × {CV_FOLDS}-fold) ---")
    cv_results = {}
    for name, factory in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        cv_results[name] = cv_score(factory, X_train, y_train, seeds=SEEDS)
        print(f"AUC {cv_results[name]['roc_auc']['mean']:.4f} ± {cv_results[name]['roc_auc']['std']:.4f}")

    # --- 5. Final held-out test evaluation ---
    print("\n--- Final test-set evaluation ---")
    test_results = {}
    for name, factory in MODELS.items():
        test_results[name] = final_test_score(factory, X_train, y_train, X_test, y_test)
        print(f"  {name}: AUC={test_results[name]['roc_auc']:.4f}  F1={test_results[name]['f1']:.4f}")

    # Leakage ceiling check
    for name, res in test_results.items():
        if res["roc_auc"] > 0.98:
            print(f"WARNING: {name} test AUC={res['roc_auc']:.4f} > 0.98 — suspect leakage!")

    # --- 6. Determine winner ---
    lr_cv_auc = cv_results["LogisticRegression"]["roc_auc"]
    gb_cv_auc = cv_results["GradientBoosting"]["roc_auc"]

    # Gap between means relative to pooled spread
    gap = gb_cv_auc["mean"] - lr_cv_auc["mean"]
    noise = max(lr_cv_auc["std"], gb_cv_auc["std"])
    # Overlap heuristic: gap < 1 std → inconclusive
    conclusive = abs(gap) > noise
    if conclusive and gap > 0:
        winner = "GradientBoosting"
        conclusion = f"Gradient Boosting outperforms Logistic Regression (gap={gap:+.4f}, noise={noise:.4f})."
    elif conclusive and gap < 0:
        winner = "LogisticRegression"
        conclusion = f"Logistic Regression outperforms Gradient Boosting (gap={gap:+.4f}, noise={noise:.4f})."
    else:
        winner = "none"
        conclusion = f"No detectable difference (gap={gap:+.4f} is within noise={noise:.4f})."

    print(f"\nConclusion: {conclusion}")

    # --- 7. Write machine-readable results ---
    os.makedirs("results", exist_ok=True)
    output = {
        "experiment_date": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": data_path,
            "n_total": len(df),
            "n_dedup_removed": dedup_removed,
            "target_rate": round(target_rate, 4),
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        "features": FEATURE_COLS,
        "dropped_as_leaky": ["account_status"],
        "sanity_checks": {
            "majority_baseline_auc": round(base["roc_auc"], 4),
            "label_shuffle_auc": round(shuffle_auc, 4),
        },
        "cv_results": cv_results,
        "test_results": test_results,
        "conclusion": {
            "winner": winner,
            "cv_auc_gap": round(gap, 4),
            "cv_auc_noise": round(noise, 4),
            "text": conclusion,
        },
    }
    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Metrics written to {metrics_path}")

    # --- 8. Write REPORT.md ---
    lr_cv = cv_results["LogisticRegression"]
    gb_cv = cv_results["GradientBoosting"]
    lr_test = test_results["LogisticRegression"]
    gb_test = test_results["GradientBoosting"]

    report = textwrap.dedent(f"""\
    # Churn Prediction: Gradient Boosting vs Logistic Regression

    **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

    ## Claim
    Does gradient boosting outperform logistic regression for predicting customer churn
    on this dataset?

    ## Design

    **Variable:** model family (LogisticRegression vs GradientBoostingClassifier).
    All other choices — features, preprocessing, split — are identical for both arms.

    **Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

    **Dropped features (with justification):**
    - `account_status`: derived directly from the target label (`"closed"` iff `churned=1`).
      Including it would be perfect label leakage.
    - `signup_date`: used only to establish split order; not predictive at inference time
      as-is (raw date encodes cohort, not individual risk).
    - `customer_id`: identifier, not a feature.

    **Preprocessing:** StandardScaler fitted on train only, applied to test.

    **Deduplication:** {dedup_removed} exact-duplicate rows removed before splitting
    (original dataset contained planted duplicates that would straddle splits).

    **Split policy:** Time-based 80/20 split on `signup_date`.
    Train: {train_df['signup_date'].min().date()} – {train_df['signup_date'].max().date()} (n={len(train_df)})
    Test:  {test_df['signup_date'].min().date()} – {test_df['signup_date'].max().date()} (n={len(test_df)})
    A random split on temporal data would leak future information into training.

    **Evaluation:** Primary metric is ROC-AUC (handles class imbalance; target rate = {target_rate:.1%}).
    CV: {len(SEEDS)} seeds × {CV_FOLDS}-fold stratified CV on the training set.
    Each seed produces {CV_FOLDS} fold scores; total n={len(SEEDS)*CV_FOLDS} per metric per model.

    ## Sanity Checks

    | Check | Value | Pass? |
    |-------|-------|-------|
    | Majority-class baseline AUC | {base['roc_auc']:.4f} | — (floor) |
    | Label-shuffle AUC (LR) | {shuffle_auc:.4f} | {'✓ near 0.5' if shuffle_auc < 0.6 else '✗ FAIL — suspect leakage'} |

    ## Results

    ### Cross-Validated Scores (train set, {len(SEEDS)} seeds × {CV_FOLDS}-fold)

    | Model | ROC-AUC | F1 | Precision | Recall |
    |-------|---------|----|-----------|--------|
    | LogisticRegression | {fmt_metric(lr_cv['roc_auc'])} | {fmt_metric(lr_cv['f1'])} | {fmt_metric(lr_cv['precision'])} | {fmt_metric(lr_cv['recall'])} |
    | GradientBoosting   | {fmt_metric(gb_cv['roc_auc'])} | {fmt_metric(gb_cv['f1'])} | {fmt_metric(gb_cv['precision'])} | {fmt_metric(gb_cv['recall'])} |

    ### Final Held-Out Test Scores (test set, single run)

    | Model | ROC-AUC | F1 | Precision | Recall |
    |-------|---------|----|-----------|--------|
    | LogisticRegression | {lr_test['roc_auc']:.4f} | {lr_test['f1']:.4f} | {lr_test['precision']:.4f} | {lr_test['recall']:.4f} |
    | GradientBoosting   | {gb_test['roc_auc']:.4f} | {gb_test['f1']:.4f} | {gb_test['precision']:.4f} | {gb_test['recall']:.4f} |

    ## Conclusion

    **{conclusion}**

    CV AUC gap: {gap:+.4f} (noise threshold: {noise:.4f}).
    {'The gap exceeds the noise floor, supporting a conclusion of difference.' if conclusive else 'The gap is within the noise floor; the honest conclusion is no detectable difference.'}

    ## Limitations

    1. **Single dataset / single seed for final test:** The held-out test result uses one
       seed (42) for the final fit; different seeds could shift results within the CV spread.
    2. **No hyperparameter tuning:** Both models use default/fixed hyperparameters.
       Tuning GB more aggressively might widen the gap further, but tuning budget must be
       equal across arms to be fair.
    3. **Temporal split by signup_date:** The split is a reasonable proxy for real deployment
       (train on early cohorts, predict for later ones), but signup_date is not the same as
       the date a churn event would be predicted in production.
    4. **Small feature set:** Only three numeric features are used. Real churn models often
       include behavioral sequences, product usage, etc.
    5. **Synthetic data:** The dataset was generated from a known logistic model; results
       may not generalise to real customer churn data.
    """)

    with open("REPORT.md", "w") as f:
        f.write(report)
    print("Report written to REPORT.md")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found. Run: python3 make_dataset.py --out {args.data}")
        sys.exit(1)

    main(args.data)
