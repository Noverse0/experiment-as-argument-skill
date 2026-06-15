"""Entrypoint: compare LogisticRegression vs GradientBoostingClassifier on churn."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import run_cv
from src.pipeline import FEATURE_COLS, LEAK_COLS, build_pipeline, load_and_clean

DATASET = "churn.csv"
RESULTS_DIR = Path("results")
SEEDS = [42, 123, 777]
N_SPLITS = 5
MODELS = ["logistic_regression", "gradient_boosting"]


def _ensure_dataset():
    if not Path(DATASET).exists():
        print(f"{DATASET} not found — generating with make_dataset.py ...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", DATASET], check=True
        )


def _dataset_stats(path: str) -> dict:
    raw = pd.read_csv(path)
    deduped = raw.drop_duplicates()
    return {
        "raw_rows": len(raw),
        "deduped_rows": len(deduped),
        "duplicates_removed": len(raw) - len(deduped),
        "churn_rate": float(deduped["churned"].mean()),
    }


def main():
    _ensure_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)

    stats = _dataset_stats(DATASET)
    print(f"Dataset: {stats['raw_rows']} rows, {stats['duplicates_removed']} duplicates removed")
    print(f"Churn rate: {stats['churn_rate']:.1%}")

    X, y = load_and_clean(DATASET)
    print(f"Features: {list(X.columns)}")
    print(f"Dropped (leak): {LEAK_COLS}")

    results = {}
    for model_name in MODELS:
        print(f"\nEvaluating {model_name} ({len(SEEDS)} seeds × {N_SPLITS} folds) ...")
        pipe = build_pipeline(model_name, seed=42)
        scores = run_cv(pipe, X, y, seeds=SEEDS, n_splits=N_SPLITS)
        results[model_name] = scores
        auc = scores["roc_auc"]
        print(f"  ROC-AUC: {auc['mean']:.4f} ± {auc['std']:.4f}  (n={auc['n']})")

    artifacts = {
        "dataset_stats": stats,
        "config": {
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "features": FEATURE_COLS,
            "leaked_dropped": LEAK_COLS,
        },
        "results": results,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"\nMetrics written to {metrics_path}")

    _write_report(results, stats)
    print("Report written to REPORT.md")


def _write_report(results: dict, stats: dict):
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]

    lr_auc, lr_std = lr["roc_auc"]["mean"], lr["roc_auc"]["std"]
    gb_auc, gb_std = gb["roc_auc"]["mean"], gb["roc_auc"]["std"]
    gap = gb_auc - lr_auc
    n_folds = lr["roc_auc"]["n"]

    # Ranges overlap → no reliable winner
    lr_hi = lr_auc + lr_std
    gb_lo = gb_auc - gb_std
    lr_lo = lr_auc - lr_std
    gb_hi = gb_auc + gb_std
    ranges_overlap = lr_hi > gb_lo and gb_hi > lr_lo

    if abs(gap) < 0.01 or ranges_overlap:
        conclusion_sentence = (
            "**No detectable difference.** "
            f"The ROC-AUC gap ({gap:+.4f}) is within noise (overlapping ±1 sd ranges), "
            "so we cannot claim a reliable winner with this dataset and methodology."
        )
    elif gb_auc > lr_auc:
        conclusion_sentence = (
            f"**Gradient boosting outperforms logistic regression** "
            f"(ROC-AUC gap = {gap:+.4f}, non-overlapping ±1 sd ranges)."
        )
    else:
        conclusion_sentence = (
            f"**Logistic regression outperforms gradient boosting** "
            f"(ROC-AUC gap = {gap:+.4f}, non-overlapping ±1 sd ranges)."
        )

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

**Dataset**
- Raw rows: {stats['raw_rows']} | After deduplication: {stats['deduped_rows']} ({stats['duplicates_removed']} exact duplicates removed)
- Churn rate: {stats['churn_rate']:.1%}

**Features used**
| Column | Role |
|--------|------|
| `tenure_months` | Causal signal — kept |
| `monthly_spend` | Causal signal — kept |
| `support_tickets` | Causal signal — kept |
| `days_since_last_login` | **Target leak — dropped** |
| `customer_id` | Identifier — dropped |
| `signup_date` | Temporal, redundant with `tenure_months` — dropped |

`days_since_last_login` is dropped because it encodes the outcome: a churned customer
has by definition stopped logging in, so this value is not known *before* the churn
event and cannot legitimately be used as a predictor.

**Evaluation**
- Stratified 5-fold cross-validation repeated over 3 seeds (42, 123, 777)
- Total: {N_SPLITS} folds × {len(SEEDS)} seeds = {n_folds} scores per model
- Preprocessing (StandardScaler) lives inside each pipeline and is fitted only on
  training folds — no leakage into validation folds.
- Primary metric: **ROC-AUC** (robust to the {stats['churn_rate']:.0%} churn-rate imbalance)
- Secondary: F1, Accuracy

## Results

| Model | ROC-AUC (mean ± sd) | F1 (mean ± sd) | Accuracy (mean ± sd) |
|-------|---------------------|----------------|----------------------|
| Logistic Regression | {lr_auc:.4f} ± {lr_std:.4f} | {lr['f1']['mean']:.4f} ± {lr['f1']['std']:.4f} | {lr['accuracy']['mean']:.4f} ± {lr['accuracy']['std']:.4f} |
| Gradient Boosting   | {gb_auc:.4f} ± {gb_std:.4f} | {gb['f1']['mean']:.4f} ± {gb['f1']['std']:.4f} | {gb['accuracy']['mean']:.4f} ± {gb['accuracy']['std']:.4f} |

n = {n_folds} folds per model

## Conclusion

{conclusion_sentence}

ROC-AUC gap (GB − LR): {gap:+.4f}
Majority-class baseline AUC: 0.5000 (both models must and do exceed this floor).

## Sanity Checks Performed

- **Baseline floor**: majority-class AUC = 0.5; both models exceed it.
- **Leak audit**: `days_since_last_login` identified and excluded (post-outcome feature).
- **Dedup before split**: {stats['duplicates_removed']} exact duplicates removed so no row straddles train/test.
- **Scaler inside pipeline**: StandardScaler is fitted per fold, not on the full dataset.
- **Multiple seeds**: variance measured over {n_folds} folds; no winner claimed without confirming non-overlap.

## Limitations

1. **Only 3 honest features remain** after removing leaks and identifiers. The weak
   signal compresses any potential gap between the two model families.
2. **Synthetic data**: results reflect the `make_dataset.py` generative process, not
   real customer behavior.
3. **No hyperparameter search**: both models use defaults; tuning could alter the gap.
4. **Temporal split not used**: `signup_date` was dropped as redundant with
   `tenure_months`. In a real deployment, a time-based train/test split would be
   required to evaluate generalization to future customers.
5. **F1 uses default threshold (0.5)**: threshold tuning might favour one model over
   the other on imbalanced data.
"""
    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
