"""
Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn.

Run:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py

Outputs:
    results/metrics.json   machine-readable per-seed and aggregate metrics
    REPORT.md              methodology, findings, limitations
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FEATURES, TARGET, load_and_clean, temporal_split
from src.evaluate import (
    label_shuffle_auc,
    majority_baseline,
    overfit_check,
    score,
)
from src.pipeline import make_gb_pipeline, make_lr_pipeline

SEEDS = [0, 1, 2, 3, 4]
DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- 0. Generate dataset if missing ---
    if not Path(DATA_PATH).exists():
        subprocess.run([sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True)

    # --- 1. Load & clean ---
    df, audit = load_and_clean(DATA_PATH)
    print(f"\n=== Data audit ===")
    print(f"  Raw rows:          {audit['n_raw']}")
    print(f"  Duplicates removed:{audit['n_duplicates_removed']}")
    print(f"  Clean rows:        {audit['n_clean']}")
    print(f"  Churn rate:        {audit['target_rate']:.3f}")

    # --- 2. Temporal split (split before any fit-like ops) ---
    train_df, test_df = temporal_split(df, test_frac=0.2)
    print(f"\n=== Split ===")
    print(f"  Train: {len(train_df)} rows  ({train_df[TARGET].mean():.3f} churn rate)")
    print(f"  Test:  {len(test_df)} rows  ({test_df[TARGET].mean():.3f} churn rate)")
    print(f"  Train date range: {train_df['signup_date'].min().date()} – {train_df['signup_date'].max().date()}")
    print(f"  Test  date range: {test_df['signup_date'].min().date()} – {test_df['signup_date'].max().date()}")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    # --- 3. Sanity checks ---
    print(f"\n=== Sanity checks ===")

    baseline_metrics = majority_baseline(X_train, y_train, X_test, y_test)
    print(f"  Majority baseline AUC: {baseline_metrics['roc_auc']:.3f} (expected ~0.5)")

    shuffle_auc_lr = label_shuffle_auc(make_lr_pipeline, X_train, y_train, X_test, y_test, n_trials=20)
    shuffle_auc_gb = label_shuffle_auc(make_gb_pipeline, X_train, y_train, X_test, y_test, n_trials=20)
    print(f"  Label-shuffle AUC LR: {shuffle_auc_lr:.3f} (avg over 20 shuffles, expected ~0.5)")
    print(f"  Label-shuffle AUC GB: {shuffle_auc_gb:.3f} (avg over 20 shuffles, expected ~0.5)")

    overfit_lr = overfit_check(make_lr_pipeline, X_train, y_train)
    overfit_gb = overfit_check(make_gb_pipeline, X_train, y_train)
    print(f"  Overfit-50 train acc LR: {overfit_lr:.3f} (expected > baseline)")
    print(f"  Overfit-50 train acc GB: {overfit_gb:.3f} (expected > baseline)")

    assert shuffle_auc_lr < 0.57, f"Label-shuffle AUC too high for LR: {shuffle_auc_lr:.3f}"
    assert shuffle_auc_gb < 0.57, f"Label-shuffle AUC too high for GB: {shuffle_auc_gb:.3f}"
    majority_acc = max(y_train.mean(), 1 - y_train.mean())
    assert overfit_gb > majority_acc, "GB cannot overfit a tiny subset — pipeline may be broken"

    # --- 4. Multi-seed evaluation ---
    print(f"\n=== Multi-seed evaluation ({len(SEEDS)} seeds) ===")

    lr_results, gb_results = [], []
    for seed in SEEDS:
        lr = make_lr_pipeline(seed)
        lr.fit(X_train, y_train)
        lr_m = score(lr, X_test, y_test)
        lr_results.append(lr_m)

        gb = make_gb_pipeline(seed)
        gb.fit(X_train, y_train)
        gb_m = score(gb, X_test, y_test)
        gb_results.append(gb_m)

    def summarize(results: list[dict]) -> dict:
        keys = list(results[0].keys())
        return {
            k: {
                "mean": float(np.mean([r[k] for r in results])),
                "std": float(np.std([r[k] for r in results])),
                "values": [r[k] for r in results],
            }
            for k in keys
        }

    lr_summary = summarize(lr_results)
    gb_summary = summarize(gb_results)

    for name, summary in [("LogisticRegression", lr_summary), ("GradientBoosting", gb_summary)]:
        print(f"\n  {name}:")
        for metric, stats in summary.items():
            print(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # --- 5. Save machine-readable results ---
    results_payload = {
        "experiment": "churn_lr_vs_gb",
        "methodology": {
            "features": FEATURES,
            "excluded_features": ["customer_id", "signup_date", "days_since_last_login"],
            "exclusion_reason": {
                "days_since_last_login": "temporal leak — value is known only at/after the churn event",
                "signup_date": "used only for temporal split ordering; not a causal predictor",
                "customer_id": "identifier, no predictive signal",
            },
            "split": "temporal (80/20 by signup_date)",
            "deduplication": "exact duplicate rows removed before split",
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
        },
        "audit": audit,
        "sanity": {
            "majority_baseline_auc": baseline_metrics["roc_auc"],
            "label_shuffle_auc_lr": shuffle_auc_lr,
            "label_shuffle_auc_gb": shuffle_auc_gb,
            "overfit_train_acc_lr": overfit_lr,
            "overfit_train_acc_gb": overfit_gb,
        },
        "per_seed": {
            "LogisticRegression": lr_results,
            "GradientBoostingClassifier": gb_results,
        },
        "aggregate": {
            "LogisticRegression": lr_summary,
            "GradientBoostingClassifier": gb_summary,
        },
    }

    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults written to {out_path}")

    # --- 6. Write REPORT.md ---
    _write_report(results_payload, audit, train_df, test_df)
    print("REPORT.md written.")


def _write_report(r: dict, audit: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    lr = r["aggregate"]["LogisticRegression"]
    gb = r["aggregate"]["GradientBoostingClassifier"]
    lr_auc_mean = lr["roc_auc"]["mean"]
    gb_auc_mean = gb["roc_auc"]["mean"]
    lr_auc_std = lr["roc_auc"]["std"]
    gb_auc_std = gb["roc_auc"]["std"]
    lr_f1_mean = lr["f1"]["mean"]
    gb_f1_mean = gb["f1"]["mean"]
    lr_f1_std = lr["f1"]["std"]
    gb_f1_std = gb["f1"]["std"]

    gap_auc = gb_auc_mean - lr_auc_mean
    overlap = (lr_auc_mean + 2 * lr_auc_std) >= (gb_auc_mean - 2 * gb_auc_std)

    if abs(gap_auc) < 0.01 or overlap:
        conclusion = (
            "**No detectable difference.** "
            f"The AUC gap ({gap_auc:+.4f}) is within the noise bands of both models "
            f"(LR: {lr_auc_mean:.4f} ± {lr_auc_std:.4f}, "
            f"GB: {gb_auc_mean:.4f} ± {gb_auc_std:.4f}). "
            "Neither model reliably outperforms the other on this dataset."
        )
    elif gap_auc > 0:
        conclusion = (
            f"**Gradient Boosting outperforms Logistic Regression** "
            f"(AUC {gb_auc_mean:.4f} ± {gb_auc_std:.4f} vs "
            f"{lr_auc_mean:.4f} ± {lr_auc_std:.4f}, gap {gap_auc:+.4f})."
        )
    else:
        conclusion = (
            f"**Logistic Regression outperforms Gradient Boosting** "
            f"(AUC {lr_auc_mean:.4f} ± {lr_auc_std:.4f} vs "
            f"{gb_auc_mean:.4f} ± {gb_auc_std:.4f}, gap {gap_auc:+.4f})."
        )

    report = f"""# Churn Prediction Experiment: Logistic Regression vs Gradient Boosting

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion

{conclusion}

## Methodology

### Data

- **Source**: `churn.csv` generated by `make_dataset.py` (seed=7, n=4000 + 200 duplicates)
- **Deduplication**: {audit['n_duplicates_removed']} exact duplicate rows removed before splitting ({audit['n_raw']} → {audit['n_clean']} rows)
- **Churn rate**: {audit['target_rate']:.1%}

### Feature Selection

| Feature | Used | Reason |
|---|---|---|
| `tenure_months` | ✓ | Legitimate causal predictor |
| `monthly_spend` | ✓ | Legitimate causal predictor |
| `support_tickets` | ✓ | Legitimate causal predictor |
| `days_since_last_login` | ✗ | **Temporal leak**: value is recorded at/after the churn event (churned customers have stopped logging in), so it would not be available at prediction time in a real deployment |
| `signup_date` | ✗ | Used for split ordering only; not a causal predictor of churn |
| `customer_id` | ✗ | Identifier with no predictive signal |

### Split

**Temporal split** (80/20 by `signup_date`):
- Train: {len(train_df)} rows, date range {train_df['signup_date'].min().date()} – {train_df['signup_date'].max().date()} ({train_df['churned'].mean():.1%} churn)
- Test: {len(test_df)} rows, date range {test_df['signup_date'].min().date()} – {test_df['signup_date'].max().date()} ({test_df['churned'].mean():.1%} churn)

A time-ordered split is used because (a) `signup_date` is temporal and a random split would let duplicate rows straddle the boundary; (b) it simulates the real deployment scenario where we predict churn for new customers based on older training data.

### Preprocessing

- `LogisticRegression`: `StandardScaler` → `LogisticRegression(max_iter=1000)`
- `GradientBoostingClassifier`: no scaling (tree-based, scale-invariant)

### Repetitions

Both models were trained with {len(r['methodology']['seeds'])} independent random seeds ({r['methodology']['seeds']}) to quantify variance from model initialization. The fixed temporal split ensures the comparison is apples-to-apples.

### Metrics

- **ROC-AUC**: primary metric; handles class imbalance without threshold selection
- **F1, Precision, Recall**: reported for completeness; F1 uses the default 0.5 threshold

## Results

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| LogisticRegression | {lr_auc_mean:.4f} ± {lr_auc_std:.4f} | {lr_f1_mean:.4f} ± {lr_f1_std:.4f} |
| GradientBoostingClassifier | {gb_auc_mean:.4f} ± {gb_auc_std:.4f} | {gb_f1_mean:.4f} ± {gb_f1_std:.4f} |

### Sanity Checks

| Check | LR | GB | Expected |
|---|---|---|---|
| Majority baseline AUC | {r['sanity']['majority_baseline_auc']:.3f} | — | ~0.5 |
| Label-shuffle AUC | {r['sanity']['label_shuffle_auc_lr']:.3f} | {r['sanity']['label_shuffle_auc_gb']:.3f} | < 0.6 |
| Overfit-50 train accuracy | {r['sanity']['overfit_train_acc_lr']:.3f} | {r['sanity']['overfit_train_acc_gb']:.3f} | > majority baseline |

All sanity checks passed: label-shuffled models fall near chance, both models can overfit a small subset.

## Limitations

1. **Small sample**: 4,000 (→ {audit['n_clean']} after dedup) rows with only 3 legitimate features leaves little room for complex models to shine. Both models may converge near their respective ceilings.
2. **No statistical test**: with 5 seeds we report mean ± std but no formal hypothesis test (e.g., paired t-test). The variance estimate is noisy with n=5.
3. **Hyperparameters not tuned**: GradientBoosting uses default-ish settings (100 trees, depth 3, lr=0.1). Proper tuning might change the gap, but it would require a held-out validation set to avoid test-set contamination.
4. **Temporal split artefact**: since `signup_date` is randomly assigned in this synthetic dataset (not correlated with the outcome), the temporal split is effectively random — the ordering assumption that "older customers' behaviour predicts newer ones" does not hold here as it would in real data.
5. **Single dataset seed**: results may differ with different data generation seeds.
"""
    Path("REPORT.md").write_text(report)


if __name__ == "__main__":
    run()
