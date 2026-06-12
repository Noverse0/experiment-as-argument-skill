"""Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn prediction.

Run:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py
"""

import json
import os
import sys

import numpy as np

from src.data import load, clean, get_X_y
from src.evaluate import cv_evaluate, summarize
from src.models import make_gb_pipeline, make_lr_pipeline

DATA_PATH = "churn.csv"
RESULTS_DIR = "results"
N_SPLITS = 5
SEEDS = [42, 123, 456]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found: {DATA_PATH}. Run: python3 make_dataset.py --out {DATA_PATH}")

    df, stats = clean(load(DATA_PATH))
    print(f"[data] {stats['n_rows']} rows after deduplication "
          f"({stats['n_duplicates_removed']} duplicates removed), "
          f"churn rate {stats['churn_rate']:.3f}")

    X, y = get_X_y(df)

    raw: dict = {"lr": [], "gb": []}

    for seed in SEEDS:
        print(f"\n[seed={seed}]")
        for name, pipeline in [("lr", make_lr_pipeline(seed)), ("gb", make_gb_pipeline(seed))]:
            fold_results = cv_evaluate(pipeline, X, y, n_splits=N_SPLITS)
            raw[name].extend(fold_results)
            aucs = [r["roc_auc"] for r in fold_results]
            print(f"  {name}: ROC-AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    summaries = {name: summarize(results) for name, results in raw.items()}

    metrics_payload = {
        "config": {"seeds": SEEDS, "n_splits": N_SPLITS, "data_path": DATA_PATH},
        "data_stats": stats,
        "summaries": summaries,
        "raw_folds": raw,
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    _write_report(summaries, stats, n_splits=N_SPLITS, seeds=SEEDS)

    lr_auc = summaries["lr"]["roc_auc"]
    gb_auc = summaries["gb"]["roc_auc"]
    gap = gb_auc["mean"] - lr_auc["mean"]
    combined_spread = lr_auc["std"] + gb_auc["std"]

    print(f"\nLR  ROC-AUC = {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}")
    print(f"GB  ROC-AUC = {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}")
    if abs(gap) < combined_spread:
        print("Conclusion: no detectable difference (gap within combined spread).")
    elif gap > 0:
        print(f"Conclusion: gradient boosting wins by {gap:.4f} ROC-AUC.")
    else:
        print(f"Conclusion: logistic regression wins by {-gap:.4f} ROC-AUC.")
    print(f"\nResults: {metrics_path}  |  Report: REPORT.md")


def _write_report(summaries: dict, stats: dict, n_splits: int, seeds: list[int]) -> None:
    lr = summaries["lr"]
    gb = summaries["gb"]
    lr_auc = lr["roc_auc"]
    gb_auc = gb["roc_auc"]
    gap = gb_auc["mean"] - lr_auc["mean"]
    combined_spread = lr_auc["std"] + gb_auc["std"]
    n_obs = len(seeds) * n_splits

    if abs(gap) < combined_spread:
        conclusion = (
            "**No detectable difference.** The gap between models "
            f"({abs(gap):.4f} ROC-AUC) is smaller than the combined spread "
            f"({combined_spread:.4f}). Neither model is a clear winner on this dataset."
        )
    elif gap > 0:
        conclusion = (
            f"**Gradient Boosting outperforms Logistic Regression** by {gap:.4f} ROC-AUC "
            f"(gap {gap:.4f} > combined spread {combined_spread:.4f})."
        )
    else:
        conclusion = (
            f"**Logistic Regression outperforms Gradient Boosting** by {-gap:.4f} ROC-AUC "
            f"(gap {-gap:.4f} > combined spread {combined_spread:.4f})."
        )

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

{conclusion}

| Model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | F1 (mean ± sd) | n |
|---|---|---|---|---|
| Logistic Regression | {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f} | {lr['pr_auc']['mean']:.4f} ± {lr['pr_auc']['std']:.4f} | {lr['f1']['mean']:.4f} ± {lr['f1']['std']:.4f} | {n_obs} |
| Gradient Boosting | {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} | {gb['pr_auc']['mean']:.4f} ± {gb['pr_auc']['std']:.4f} | {gb['f1']['mean']:.4f} ± {gb['f1']['std']:.4f} | {n_obs} |

n = {len(seeds)} seeds × {n_splits} CV folds = {n_obs} fold-seed observations per model.

## Methodology

**Claim:** Does gradient boosting outperform logistic regression for predicting customer churn?

**Single variable:** Model family. All other factors are held fixed (features, preprocessing,
CV scheme, random seeds).

**Data:** {stats['n_rows']} rows after deduplication. Churn rate: {stats['churn_rate']:.3f}.

### Leakage mitigations

| Issue | Action taken |
|---|---|
| `account_status` encodes the label (closed ↔ churned=1) | Dropped before training |
| 200 exact duplicate rows appended to the dataset | Deduplicated before any split |
| `customer_id` is a row identifier | Dropped before training |
| `signup_date` is temporal; random splits would leak future signal | Temporal split used (see below) |

### Split policy

`TimeSeriesSplit(n_splits={n_splits})` on rows sorted ascending by `signup_date`.
Each fold trains on earlier-signup customers and evaluates on later ones — mirroring
production usage where you score customers who signed up more recently than your
training data. This prevents future leakage that a random split would introduce on
time-ordered data.

### Features

`tenure_months`, `monthly_spend`, `support_tickets`.
`signup_date` informs split order only; it is not passed to the model because its
information is already captured by `tenure_months`.

### Preprocessing

- **Logistic Regression:** `StandardScaler` fitted on each training fold, applied to the
  corresponding test fold. Required so that regularisation acts uniformly across features.
- **Gradient Boosting:** no scaling (tree-based splits are scale-invariant).

### Runs

{len(seeds)} random seeds × {n_splits} CV folds = {n_obs} observations per arm. Seeds
vary model-internal randomness (tree construction for GB, solver tie-breaking for LR).

### Primary metric

ROC-AUC: threshold-free, robust to the class imbalance present in this dataset
(churn rate {stats['churn_rate']:.3f}).

## Limitations

- **Small feature set (3 features).** Richer features might shift the comparison.
- **Default / lightly tuned hyperparameters.** A matched tuning budget per arm could
  alter results; the current comparison favours neither.
- **Synthetic data.** The generative process is a simple logistic model; gradient boosting
  has no advantage over logistic regression on a linearly separable signal.
- **{n_obs} fold-seeds is modest.** The spread (sd) should be treated as indicative;
  formal statistical significance would require more replications.
- **Negative / null results are reported as-is.** No post-hoc selection of seeds or folds.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
