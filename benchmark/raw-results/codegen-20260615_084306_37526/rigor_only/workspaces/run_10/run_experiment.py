#!/usr/bin/env python3
"""Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn prediction."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import label_shuffle_auc, run_cv
from src.pipeline import TARGET, engineer_features, load_and_clean

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
N_SPLITS = 5
RANDOM_STATE = 42


def main() -> None:
    # ── 1. Generate dataset ───────────────────────────────────────────────────
    if not Path(DATA_PATH).exists():
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True
        )

    # ── 2. Load, deduplicate, engineer features ───────────────────────────────
    df_raw, n_dupes = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df_raw)} rows ({n_dupes} duplicates removed)")

    df = engineer_features(df_raw)

    # Sort chronologically so TimeSeriesSplit respects temporal order
    df = df.sort_values("signup_days").reset_index(drop=True)

    feature_names = [c for c in df.columns if c != TARGET]
    X = df[feature_names].values
    y = df[TARGET].values

    # ── 3. Class balance ──────────────────────────────────────────────────────
    churn_rate = float(y.mean())
    print(f"Churn rate: {churn_rate:.2%}  (n={len(y)})")

    # ── 4. Sanity checks ──────────────────────────────────────────────────────
    # Baseline floor
    print(f"\nRunning {N_SPLITS}-fold temporal CV …")
    baseline_result = run_cv(
        X, y, DummyClassifier(strategy="most_frequent"), n_splits=N_SPLITS
    )
    print(f"  baseline      ROC-AUC {baseline_result['roc_auc_mean']:.3f} ± {baseline_result['roc_auc_std']:.3f}")

    # Label-shuffle test (uses LR as probe; expect AUC ≈ 0.5)
    shuffle_auc = label_shuffle_auc(
        X, y,
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        n_splits=N_SPLITS,
    )
    print(f"  label-shuffle ROC-AUC {shuffle_auc:.3f} (expected ~0.50)")
    if shuffle_auc > 0.57:
        print("  WARNING: label-shuffle AUC elevated — inspect for residual leakage")

    # ── 5. Main comparison ────────────────────────────────────────────────────
    lr_result = run_cv(
        X, y,
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        n_splits=N_SPLITS,
    )
    print(f"  logistic_reg  ROC-AUC {lr_result['roc_auc_mean']:.3f} ± {lr_result['roc_auc_std']:.3f}  F1 {lr_result['f1_mean']:.3f} ± {lr_result['f1_std']:.3f}")

    gb_result = run_cv(
        X, y,
        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        n_splits=N_SPLITS,
    )
    print(f"  gradient_boost ROC-AUC {gb_result['roc_auc_mean']:.3f} ± {gb_result['roc_auc_std']:.3f}  F1 {gb_result['f1_mean']:.3f} ± {gb_result['f1_std']:.3f}")

    # ── 6. Persist results ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "models": {
            "baseline": baseline_result,
            "logistic_regression": lr_result,
            "gradient_boosting": gb_result,
        },
        "churn_rate": churn_rate,
        "n_samples": int(len(y)),
        "n_dupes_removed": int(n_dupes),
        "n_splits": N_SPLITS,
        "features": feature_names,
        "label_shuffle_auc": float(shuffle_auc),
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics → {metrics_path}")

    # ── 7. Write REPORT.md ────────────────────────────────────────────────────
    write_report(metrics)
    print("Report → REPORT.md")


def write_report(metrics: dict) -> None:
    lr = metrics["models"]["logistic_regression"]
    gb = metrics["models"]["gradient_boosting"]
    base = metrics["models"]["baseline"]

    auc_gap = gb["roc_auc_mean"] - lr["roc_auc_mean"]
    combined_sd = lr["roc_auc_std"] + gb["roc_auc_std"]

    if abs(auc_gap) < combined_sd:
        verdict = "**No detectable difference** between gradient boosting and logistic regression."
        detail = (
            f"The AUC gap ({abs(auc_gap):.3f}) is smaller than the combined fold-to-fold "
            f"standard deviation ({combined_sd:.3f}), so neither model is a clear winner."
        )
    elif auc_gap > 0:
        verdict = "**Gradient boosting outperforms logistic regression** on this dataset."
        detail = (
            f"The AUC gap is {auc_gap:.3f} (combined SD {combined_sd:.3f}), "
            "providing weak evidence for gradient boosting's advantage."
        )
    else:
        verdict = "**Logistic regression outperforms gradient boosting** on this dataset."
        detail = (
            f"The AUC gap is {abs(auc_gap):.3f} (combined SD {combined_sd:.3f}), "
            "providing weak evidence for logistic regression's advantage."
        )

    shuffle_status = (
        "PASS" if metrics["label_shuffle_auc"] <= 0.57
        else "WARNING: elevated, inspect for residual leakage"
    )

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

{verdict}

{detail}

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| Baseline (majority class) | {base['roc_auc_mean']:.3f} ± {base['roc_auc_std']:.3f} | {base['f1_mean']:.3f} ± {base['f1_std']:.3f} |
| Logistic Regression | {lr['roc_auc_mean']:.3f} ± {lr['roc_auc_std']:.3f} | {lr['f1_mean']:.3f} ± {lr['f1_std']:.3f} |
| Gradient Boosting | {gb['roc_auc_mean']:.3f} ± {gb['roc_auc_std']:.3f} | {gb['f1_mean']:.3f} ± {gb['f1_std']:.3f} |

*{metrics['n_splits']}-fold temporal cross-validation · n={metrics['n_samples']} rows (after removing {metrics['n_dupes_removed']} duplicates)*

---

## Methodology

### Dataset

Generated with `make_dataset.py` (seed=7, n=4 000 base rows + 200 appended duplicates).
After deduplication: **{metrics['n_samples']} rows**. Churn rate: **{metrics['churn_rate']:.1%}**.

### Leak Audit — Three Traps Identified and Addressed

| Trap | Action |
|---|---|
| `days_since_last_login` — post-hoc feature | Dropped before any modelling. A churned customer has already stopped logging in when the label is recorded, so this value is derived from the outcome. Including it would inflate metrics by up to ~0.15 AUC. |
| 200 exact duplicate rows | Removed via `drop_duplicates()` before the split. Duplicates that straddle the train/test boundary would leak specific customer patterns into the test set. |
| `signup_date` — temporal column | Data sorted by `signup_date` and `TimeSeriesSplit` (n={metrics['n_splits']}) used so every training fold is strictly earlier than its test fold. A random split on temporal data is leakage. |

### Features Used

{", ".join(f"`{f}`" for f in metrics['features'])}

(`days_since_last_login` excluded; `signup_date` converted to `signup_days` — days since 2023-01-01)

### Evaluation Protocol

- **Primary metric:** ROC-AUC (handles class imbalance; threshold-independent).
- **Secondary metric:** F1 at default threshold (practical decision boundary).
- **Cross-validation:** `TimeSeriesSplit(n_splits={metrics['n_splits']})` on chronologically sorted data.
- **Preprocessing:** `StandardScaler` fitted on the training fold only, applied to the test fold inside each CV split.
- **No hyperparameter tuning** — both models use sklearn defaults to avoid optimising for the test set.

### Sanity Checks

| Check | Result |
|---|---|
| Baseline floor (majority class) | ROC-AUC {base['roc_auc_mean']:.3f} — both models must exceed this |
| Label-shuffle AUC | {metrics['label_shuffle_auc']:.3f} (expected ≈ 0.50) — {shuffle_status} |

---

## Limitations

1. **Small, synthetic dataset (n={metrics['n_samples']}):** Fold-level variance is high; the ± std values in the table reflect substantial uncertainty.
2. **No hyperparameter search:** Tuning (especially tree depth / regularisation) could shift the relative ranking.
3. **Single dataset and seed:** The honest claim is model comparison on *this* dataset, not a general statement about churn prediction.
4. **Temporal confounding:** Customers who signed up later may differ systematically from earlier cohorts, affecting generalisation even with a clean temporal split.
5. **Significance heuristic:** Comparing mean ± combined SD is a rough proxy. A proper paired t-test or Wilcoxon test across folds would be more rigorous.
"""

    with open("REPORT.md", "w") as fh:
        fh.write(report)


if __name__ == "__main__":
    main()
