#!/usr/bin/env python3
"""Run the churn prediction comparison experiment.

Writes results/metrics.json and REPORT.md.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.data import load_and_prepare
from src.evaluate import compare, cv_evaluate
from src.models import make_gradient_boosting, make_logistic_regression
from src.sanity import (
    check_baseline,
    check_class_balance,
    check_label_shuffle,
    check_overfit_tiny,
)

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
N_SPLITS = 5


def main() -> None:
    # --- 0. Generate dataset if absent ---
    if not Path(DATA_PATH).exists():
        print(f"Generating {DATA_PATH} ...")
        subprocess.run([sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True)

    # --- 1. Load and prepare ---
    print("Loading and preparing data ...")
    X, y, load_info = load_and_prepare(DATA_PATH)
    print(
        f"  {len(X)} rows after deduplication "
        f"(removed {load_info['n_dropped_duplicates']} duplicates)"
    )

    lr_model = make_logistic_regression()
    gbm_model = make_gradient_boosting()

    # --- 2. Sanity checks ---
    print("\n--- Sanity checks ---")
    balance = check_class_balance(y)
    print(
        f"  Class balance: {balance['n_positive']}/{balance['n']} positive "
        f"({balance['positive_rate']:.1%})"
    )

    baseline_auc = check_baseline(X, y, n_splits=N_SPLITS)
    print(f"  Majority-class baseline AUC: {baseline_auc:.3f}")

    lr_overfit = check_overfit_tiny(X, y, lr_model)
    gbm_overfit = check_overfit_tiny(X, y, gbm_model)
    print(f"  Overfit-tiny accuracy  LR={lr_overfit:.2f}  GBM={gbm_overfit:.2f}")

    lr_shuffle = check_label_shuffle(X, y, lr_model, n_splits=N_SPLITS)
    gbm_shuffle = check_label_shuffle(X, y, gbm_model, n_splits=N_SPLITS)
    print(f"  Label-shuffle AUC      LR={lr_shuffle:.3f}  GBM={gbm_shuffle:.3f}")

    # Warn if label-shuffle AUC is suspiciously high (possible residual leak).
    for name, auc in [("LR", lr_shuffle), ("GBM", gbm_shuffle)]:
        if abs(auc - 0.5) > 0.1:
            print(f"  WARNING: {name} label-shuffle AUC={auc:.3f} — possible leak in features")

    sanity = {
        "class_balance": balance,
        "baseline_auc": baseline_auc,
        "overfit_tiny": {"lr": lr_overfit, "gbm": gbm_overfit},
        "label_shuffle_auc": {"lr": lr_shuffle, "gbm": gbm_shuffle},
        "n_dropped_duplicates": load_info["n_dropped_duplicates"],
    }

    # --- 3. Cross-validation ---
    print(f"\n--- Cross-validation (TimeSeriesSplit, {N_SPLITS} folds) ---")
    print("  Evaluating Logistic Regression ...")
    lr_results = cv_evaluate(X, y, lr_model, n_splits=N_SPLITS)

    print("  Evaluating Gradient Boosting ...")
    gbm_results = cv_evaluate(X, y, gbm_model, n_splits=N_SPLITS)

    for label, res in [("LR ", lr_results), ("GBM", gbm_results)]:
        for metric, vals in res.items():
            print(f"  {label}  {metric:<20}  {vals['mean']:.3f} ± {vals['std']:.3f}")

    winner = compare(lr_results, gbm_results, primary="roc_auc")
    print(f"\n  Winner (ROC-AUC): {winner}")

    # --- 4. Persist results ---
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "logistic_regression": lr_results,
        "gradient_boosting": gbm_results,
        "winner": winner,
        "sanity": sanity,
        "methodology": {
            "cv": f"TimeSeriesSplit(n_splits={N_SPLITS})",
            "primary_metric": "roc_auc",
            "features": list(X.columns),
            "dropped_features": {
                "days_since_last_login": "post-outcome leak",
                "customer_id": "identifier",
                "signup_date": "used only for temporal ordering and month extraction",
            },
            "n_samples": len(X),
        },
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {metrics_path}")

    _write_report(metrics)
    print("Wrote REPORT.md")


def _write_report(metrics: dict) -> None:
    lr = metrics["logistic_regression"]
    gbm = metrics["gradient_boosting"]
    winner = metrics["winner"]
    sanity = metrics["sanity"]
    method = metrics["methodology"]

    winner_sentence = {
        "gradient_boosting": (
            "**Gradient Boosting outperforms Logistic Regression** on ROC-AUC "
            f"({gbm['roc_auc']['mean']:.3f} vs {lr['roc_auc']['mean']:.3f})."
        ),
        "logistic_regression": (
            "**Logistic Regression outperforms Gradient Boosting** on ROC-AUC "
            f"({lr['roc_auc']['mean']:.3f} vs {gbm['roc_auc']['mean']:.3f})."
        ),
        "no_detectable_difference": (
            "**No detectable difference** between the two models: the gap in mean "
            f"ROC-AUC ({abs(gbm['roc_auc']['mean'] - lr['roc_auc']['mean']):.3f}) "
            "is within the noise floor (max fold std: "
            f"{max(lr['roc_auc']['std'], gbm['roc_auc']['std']):.3f})."
        ),
    }[winner]

    def _shuffle_verdict(auc: float) -> str:
        return "PASS (near 0.5)" if abs(auc - 0.5) <= 0.1 else "WARN (far from 0.5)"

    def _overfit_verdict(acc: float) -> str:
        return "PASS" if acc >= 0.9 else "MARGINAL"

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

{winner_sentence}

## Methodology

### Data

- **Source:** `churn.csv` generated by `make_dataset.py` (seed=7, n=4200 including 200 injected duplicates).
- **After deduplication:** {method['n_samples']} rows ({sanity['n_dropped_duplicates']} exact-duplicate rows removed before any split).
- **Target:** `churned` (binary). Positive rate: {sanity['class_balance']['positive_rate']:.1%} ({sanity['class_balance']['n_positive']} of {sanity['class_balance']['n']}).

### Leakage Analysis and Feature Selection

Three columns were dropped before modelling:

| Column | Reason |
|--------|--------|
| `days_since_last_login` | **Post-outcome leak.** Churned customers have, by definition, stopped logging in. This value is recorded after the churn outcome is determined, not before it. Including it would inflate AUC without reflecting predictive power available at decision time. |
| `customer_id` | Identifier with no predictive signal. |
| `signup_date` | Used only to sort records for temporal CV; `signup_month` (1–12) extracted as a seasonality proxy. |

**Features used:** `{", ".join(method['features'])}`

### Evaluation Protocol

- **Deduplication before split:** 200 exact-duplicate rows removed. A random split would let original/duplicate pairs straddle the boundary, inflating test-set performance through memorisation.
- **Temporal ordering:** Data sorted ascending by `signup_date` before cross-validation so that each test fold contains only records signed up after the corresponding training fold.
- **Cross-validation:** `TimeSeriesSplit(n_splits={method['cv'].split('=')[1][:-1]})` — each fold trains on the earliest records and tests on a strictly later window.
- **Primary metric:** ROC-AUC (invariant to class imbalance, does not require a threshold).
- **Secondary metrics:** F1 (macro), Average Precision.
- **Preprocessing:** `StandardScaler` applied inside the LR pipeline only (fitted on the training fold, applied to the test fold — no leakage). GBM requires no scaling.

### Hyperparameters (Fixed, No Tuning)

| Model | Config |
|-------|--------|
| LogisticRegression | C=1.0, max_iter=1000, solver=lbfgs |
| GradientBoostingClassifier | n_estimators=100, max_depth=3, learning_rate=0.1 |

## Sanity Checks

| Check | LR | GBM |
|-------|----|-----|
| Baseline AUC (majority class) | {sanity['baseline_auc']:.3f} | — |
| Overfit tiny set (20 samples) | {sanity['overfit_tiny']['lr']:.2f} — {_overfit_verdict(sanity['overfit_tiny']['lr'])} | {sanity['overfit_tiny']['gbm']:.2f} — {_overfit_verdict(sanity['overfit_tiny']['gbm'])} |
| Label-shuffle AUC | {sanity['label_shuffle_auc']['lr']:.3f} — {_shuffle_verdict(sanity['label_shuffle_auc']['lr'])} | {sanity['label_shuffle_auc']['gbm']:.3f} — {_shuffle_verdict(sanity['label_shuffle_auc']['gbm'])} |

## Results

### ROC-AUC (primary metric)

| Model | Mean | ± Std | Fold scores |
|-------|------|-------|-------------|
| Logistic Regression | {lr['roc_auc']['mean']:.3f} | {lr['roc_auc']['std']:.3f} | {", ".join(f"{s:.3f}" for s in lr['roc_auc']['scores'])} |
| Gradient Boosting   | {gbm['roc_auc']['mean']:.3f} | {gbm['roc_auc']['std']:.3f} | {", ".join(f"{s:.3f}" for s in gbm['roc_auc']['scores'])} |

Gap: {abs(gbm['roc_auc']['mean'] - lr['roc_auc']['mean']):.4f} | Noise floor (max std): {max(lr['roc_auc']['std'], gbm['roc_auc']['std']):.4f}

### F1

| Model | Mean | ± Std |
|-------|------|-------|
| Logistic Regression | {lr['f1']['mean']:.3f} | {lr['f1']['std']:.3f} |
| Gradient Boosting   | {gbm['f1']['mean']:.3f} | {gbm['f1']['std']:.3f} |

### Average Precision

| Model | Mean | ± Std |
|-------|------|-------|
| Logistic Regression | {lr['average_precision']['mean']:.3f} | {lr['average_precision']['std']:.3f} |
| Gradient Boosting   | {gbm['average_precision']['mean']:.3f} | {gbm['average_precision']['std']:.3f} |

## Limitations

1. **No hyperparameter tuning.** Both models use fixed defaults. A nested CV grid search might change the outcome, particularly for GBM which is sensitive to `n_estimators`, `learning_rate`, and `max_depth`.
2. **Five-fold estimate.** Variance is estimated across 5 temporal folds. This is a thin basis for statistical inference; the "no detectable difference" criterion is correspondingly conservative.
3. **Synthetic, low-signal dataset.** The DGP uses modest logistic coefficients; AUC is expected to be in the 0.55–0.70 range after removing the leaked feature. Results may not generalise to real churn datasets.
4. **Dropped temporal feature.** `signup_date` contributes only `signup_month`. Richer temporal features (e.g., cohort effects, trend) could be extracted but were excluded to avoid scope creep.
5. **Single data version.** Experiment runs on one fixed seed (7). A seed sweep across dataset generation would quantify sensitivity to sampling randomness.
"""

    Path("REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
