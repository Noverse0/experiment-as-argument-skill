"""Entrypoint: compare LogisticRegression vs GradientBoosting on churn prediction."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import load_and_clean, make_features, make_lr_pipeline, make_gb_pipeline
from evaluate import evaluate_model, summarize

DATA_PATH = "churn.csv"
RESULTS_DIR = "results"
REPORT_PATH = "REPORT.md"


def run_sanity_checks(X: pd.DataFrame, y: pd.Series, ref_pipeline) -> dict:
    """Quick checks that catch most silent pipeline bugs before full CV."""
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = next(iter(tscv.split(X)))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # 1. Baseline floor — stratified random guess
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f"  Baseline AUC (stratified random):  {baseline_auc:.3f}")

    # 2. Overfit one tiny slice — model must memorise it
    tiny_X = X_train.iloc[:50]
    tiny_y = y_train.iloc[:50]
    m = clone(ref_pipeline)
    m.fit(tiny_X, tiny_y)
    tiny_acc = (m.predict(tiny_X) == tiny_y).mean()
    print(f"  Tiny-subset train accuracy:        {tiny_acc:.3f}  (expect > 0.80)")

    # 3. Label-shuffle — AUC must fall to chance
    y_shuffled = y_train.sample(frac=1, random_state=99).values
    m_shuf = clone(ref_pipeline)
    m_shuf.fit(X_train, y_shuffled)
    shuffle_auc = roc_auc_score(y_test, m_shuf.predict_proba(X_test)[:, 1])
    print(f"  Label-shuffle AUC:                 {shuffle_auc:.3f}  (expect < 0.60)")

    return {
        "baseline_auc": float(baseline_auc),
        "tiny_train_acc": float(tiny_acc),
        "shuffle_auc": float(shuffle_auc),
    }


def write_report(results: dict, sanity: dict, class_balance: dict, n_deduped: int) -> str:
    lr = results["LogisticRegression"]
    gb = results["GradientBoosting"]
    lr_auc, lr_std = lr["roc_auc"]["mean"], lr["roc_auc"]["std"]
    gb_auc, gb_std = gb["roc_auc"]["mean"], gb["roc_auc"]["std"]
    n_folds = lr["roc_auc"]["n"]

    # ±1 std overlap is a simple indicator; with 5 folds a formal test has low power
    overlap = (gb_auc - gb_std) <= (lr_auc + lr_std) and (lr_auc - lr_std) <= (gb_auc + gb_std)
    if overlap:
        conclusion = (
            f"**No detectable difference.** Mean ROC-AUC: "
            f"GradientBoosting {gb_auc:.4f} ± {gb_std:.4f}, "
            f"LogisticRegression {lr_auc:.4f} ± {lr_std:.4f} "
            f"(±1 std intervals overlap; n={n_folds} folds)."
        )
    elif gb_auc > lr_auc:
        conclusion = (
            f"**GradientBoosting outperforms LogisticRegression** — "
            f"mean AUC {gb_auc:.4f} ± {gb_std:.4f} vs {lr_auc:.4f} ± {lr_std:.4f} "
            f"over {n_folds} folds."
        )
    else:
        conclusion = (
            f"**LogisticRegression outperforms GradientBoosting** — "
            f"mean AUC {lr_auc:.4f} ± {lr_std:.4f} vs {gb_auc:.4f} ± {gb_std:.4f} "
            f"over {n_folds} folds."
        )

    lr_vals = ", ".join(f"{v:.4f}" for v in lr["roc_auc"]["values"])
    gb_vals = ", ".join(f"{v:.4f}" for v in gb["roc_auc"]["values"])

    return f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion
{conclusion}

## Results

| Model | ROC-AUC mean ± std | F1 mean ± std | Folds |
|---|---|---|---|
| LogisticRegression | {lr_auc:.4f} ± {lr_std:.4f} | {lr["f1"]["mean"]:.4f} ± {lr["f1"]["std"]:.4f} | {n_folds} |
| GradientBoosting | {gb_auc:.4f} ± {gb_std:.4f} | {gb["f1"]["mean"]:.4f} ± {gb["f1"]["std"]:.4f} | {n_folds} |

Per-fold ROC-AUC:
- LogisticRegression: [{lr_vals}]
- GradientBoosting:   [{gb_vals}]

## Methodology

**Variable:** Model type (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, splits, evaluation metric, random seeds — are held identical.

**Dataset:** {class_balance["n_total"]} rows after deduplication
(removed {n_deduped} exact duplicate rows appended in the source generator).
Target rate: {class_balance["churn_rate"]:.1%} positive (churned).

**Data discipline:**
- `account_status` dropped — it is derived directly from `churned` (perfect leakage trap in the generator).
- `customer_id` dropped — identifier with no predictive signal.
- {n_deduped} exact duplicate rows removed before any split to prevent train/test contamination.
- `signup_date` converted to `signup_age_days` (days since earliest signup); the raw date column is discarded.
- No fit-like transform (StandardScaler) touches test data; scaling is inside the Pipeline and is fitted only on each fold's training split.

**Evaluation:** `TimeSeriesSplit(n_splits=5)` on data sorted by `signup_date`.
Each fold trains on earlier customers and evaluates on later ones, respecting temporal ordering.
This is the correct choice because `signup_date` is a temporal column and random splits would
allow information from future customers to leak into the training set.

**Pipelines:**
- LogisticRegression: `StandardScaler → LogisticRegression(max_iter=1000, random_state=42)`
- GradientBoosting: `GradientBoostingClassifier(n_estimators=100, random_state=42)` (no scaling needed)

**Primary metric:** ROC-AUC — robust to class imbalance and threshold-independent.
F1 reported as secondary.

## Sanity Checks

| Check | Value | Status |
|---|---|---|
| Baseline AUC (stratified random) | {sanity["baseline_auc"]:.3f} | {"PASS" if sanity["baseline_auc"] < 0.6 else "WARN"} |
| Tiny-subset train accuracy | {sanity["tiny_train_acc"]:.3f} | {"PASS" if sanity["tiny_train_acc"] > 0.80 else "FAIL — pipeline may be broken"} |
| Label-shuffle AUC | {sanity["shuffle_auc"]:.3f} | {"PASS" if sanity["shuffle_auc"] < 0.60 else "WARN — possible remaining leakage"} |

## Limitations

1. **Single synthetic dataset.** The generator uses a known logistic ground truth; real churn
   datasets are messier and may favour tree-based methods differently.
2. **No hyperparameter tuning.** Default parameters used for both models; tuning could change the
   gap but would require a separate validation split.
3. **5-fold variance estimate.** With n=5 folds, ±1 std overlap is a weak substitute for a
   formal paired test. The null result should be interpreted as "no strong evidence either way",
   not "definitely equal."
4. **Increasing train sizes across folds.** TimeSeriesSplit grows the training set fold by fold,
   so later folds may favour the more data-efficient model (GB). Both models face the same regime,
   so the comparison remains internally valid.
"""


def main():
    if not os.path.exists(DATA_PATH):
        print("Generating dataset …")
        subprocess.run([sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True)

    raw = pd.read_csv(DATA_PATH)
    df = load_and_clean(DATA_PATH)
    n_deduped = len(raw) - len(df)

    df = df.sort_values("signup_date").reset_index(drop=True)
    X, y = make_features(df)

    print(f"\nDataset: {len(X)} rows × {X.shape[1]} features after cleaning")
    print(f"Features: {list(X.columns)}")
    print(f"Target rate: {y.mean():.1%} positive (churned)")

    class_balance = {"n_total": int(len(X)), "churn_rate": float(y.mean())}

    print("\n--- Sanity Checks ---")
    sanity = run_sanity_checks(X, y, make_lr_pipeline())

    print("\n--- Cross-Validation (TimeSeriesSplit, 5 folds) ---")
    models = {
        "LogisticRegression": make_lr_pipeline(),
        "GradientBoosting": make_gb_pipeline(),
    }

    results = {}
    for name, pipeline in models.items():
        print(f"  {name} …", end=" ", flush=True)
        raw_metrics = evaluate_model(pipeline, X, y, n_splits=5)
        results[name] = summarize(raw_metrics)
        auc = results[name]["roc_auc"]
        print(f"ROC-AUC {auc['mean']:.4f} ± {auc['std']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    payload = {
        "models": results,
        "sanity_checks": sanity,
        "dataset": class_balance,
        "methodology": {
            "eval": "TimeSeriesSplit",
            "n_splits": 5,
            "sort_column": "signup_date",
            "leakage_cols_dropped": ["account_status"],
            "id_cols_dropped": ["customer_id"],
            "n_duplicates_removed": int(n_deduped),
            "primary_metric": "roc_auc",
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults  → {metrics_path}")

    report = write_report(results, sanity, class_balance, n_deduped)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report   → {REPORT_PATH}")

    lr_auc = results["LogisticRegression"]["roc_auc"]["mean"]
    gb_auc = results["GradientBoosting"]["roc_auc"]["mean"]
    print(f"\n=== CONCLUSION ===")
    print(f"  LogisticRegression ROC-AUC: {lr_auc:.4f}")
    print(f"  GradientBoosting   ROC-AUC: {gb_auc:.4f}")
    print(f"  See {REPORT_PATH} for full analysis.")


if __name__ == "__main__":
    main()
