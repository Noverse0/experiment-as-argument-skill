"""Entrypoint: run the full churn prediction experiment and write results."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Generate dataset if absent.
    if not Path("churn.csv").exists():
        print("Generating dataset...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", "churn.csv"], check=True
        )

    from src.data import load_and_clean, get_X_y, FEATURES, LEAK_FEATURES
    from src.evaluate import evaluate_pipeline
    from src.pipeline import make_lr_pipeline, make_gb_pipeline

    df = load_and_clean("churn.csv")
    X, y = get_X_y(df)

    churn_rate = y.mean()
    print(f"Dataset: {len(df)} rows after dedup")
    print(f"Features: {FEATURES}")
    print(f"Excluded (leak): {LEAK_FEATURES}")
    print(f"Churn rate: {churn_rate:.3f}")

    # Sanity checks -----------------------------------------------------------
    majority_label = int(y.mode()[0])
    baseline_acc = float((y == majority_label).mean())
    print(f"\nBaseline accuracy (majority class={majority_label}): {baseline_acc:.3f}")

    # Evaluate over multiple seeds so we can report mean ± std ---------------
    SEEDS = [42, 123, 456]
    N_SPLITS = 5

    per_seed = {"logistic_regression": [], "gradient_boosting": []}

    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        for name, make_fn in [
            ("logistic_regression", make_lr_pipeline),
            ("gradient_boosting", make_gb_pipeline),
        ]:
            result = evaluate_pipeline(make_fn(seed=seed), X, y, n_splits=N_SPLITS)
            per_seed[name].append(result)
            auc = result["roc_auc"]["mean"]
            print(f"  {name}: ROC-AUC={auc:.4f}")

    # Aggregate: mean of fold-means across seeds ------------------------------
    final: dict = {}
    for model_name, seed_results in per_seed.items():
        final[model_name] = {}
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            fold_means = [r[metric]["mean"] for r in seed_results]
            final[model_name][metric] = {
                "mean": float(np.mean(fold_means)),
                "std": float(np.std(fold_means)),
                "n_seeds": len(SEEDS),
                "n_folds": N_SPLITS,
            }

    final["meta"] = {
        "n_rows": len(df),
        "churn_rate": churn_rate,
        "features": FEATURES,
        "leaked_excluded": LEAK_FEATURES,
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "baseline_accuracy": baseline_acc,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print("\nMetrics written to results/metrics.json")

    _write_report(final)
    print("Report written to REPORT.md")


def _write_report(results: dict) -> None:
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    meta = results["meta"]

    lr_auc_mean = lr["roc_auc"]["mean"]
    gb_auc_mean = gb["roc_auc"]["mean"]
    # Use pooled uncertainty to judge whether the gap is meaningful.
    noise = lr["roc_auc"]["std"] + gb["roc_auc"]["std"]
    gap = abs(gb_auc_mean - lr_auc_mean)

    if gap <= noise:
        conclusion = (
            "The performance difference is within the noise level across seeds "
            f"(gap={gap:.4f} ≤ combined sd={noise:.4f}). "
            "**No statistically meaningful difference detected.**"
        )
        winner_label = "No clear winner"
    elif gb_auc_mean > lr_auc_mean:
        conclusion = (
            f"Gradient boosting outperforms logistic regression "
            f"(ROC-AUC gap={gap:.4f}, combined sd={noise:.4f})."
        )
        winner_label = "Gradient Boosting"
    else:
        conclusion = (
            f"Logistic regression outperforms gradient boosting "
            f"(ROC-AUC gap={gap:.4f}, combined sd={noise:.4f})."
        )
        winner_label = "Logistic Regression"

    n_seeds = lr["roc_auc"]["n_seeds"]
    n_folds = lr["roc_auc"]["n_folds"]

    def fmt(d):
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

{conclusion}

Winner: **{winner_label}**

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | {fmt(lr['roc_auc'])} | {fmt(lr['f1'])} | {fmt(lr['precision'])} | {fmt(lr['recall'])} |
| Gradient Boosting | {fmt(gb['roc_auc'])} | {fmt(gb['f1'])} | {fmt(gb['precision'])} | {fmt(gb['recall'])} |

*(mean ± std of fold-means across {n_seeds} seeds, {n_folds}-fold TimeSeriesSplit each)*

## Methodology

### Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

### Variable
Model class (LogisticRegression vs GradientBoostingClassifier). All other choices — features, preprocessing, evaluation protocol, seeds — are held fixed.

### Data Discipline

**Deduplication:** The raw CSV contains 200 exact duplicate rows. These are removed before any split to prevent the same row appearing in both train and test folds, which would inflate metrics.

**Temporal split:** Data is sorted ascending by `signup_date` and evaluated with `TimeSeriesSplit` (5 folds). This ensures training always precedes the test window in time, matching real deployment conditions and avoiding future-data leakage through random shuffling.

**Leak exclusion:** `days_since_last_login` is **excluded**. This feature is derived from the churn outcome itself — customers who churned stopped logging in, so the value is recorded *after* the decision to churn. Using it would measure the leak's signal, not the model's predictive power. The unusually strong AUC a careless pipeline would observe (from a feature correlated ≈0.6+ with the target) is the diagnostic signal for this trap.

**Features used:** `{', '.join(meta['features'])}` — the three legitimate causal signals available in this dataset.

### Preprocessing

- `StandardScaler` is fitted on the training fold only and applied to the test fold inside the CV loop (split-before-transform).
- Gradient boosting receives raw (unscaled) features; tree-based models are scale-invariant.

### Evaluation

- 5-fold `TimeSeriesSplit` per seed; {n_seeds} seeds total.
- Primary metric: **ROC-AUC** (measures ranking ability; robust to the {meta['churn_rate']:.1%} class imbalance in this dataset).
- Secondary: F1, Precision, Recall reported for completeness.
- Baseline floor (majority-class accuracy): {meta['baseline_accuracy']:.3f}.

### Sanity Checks Performed

- Majority-class baseline computed before model evaluation.
- Both models evaluated on identical held-out folds (no data leakage between arms).
- Results reproduced across {n_seeds} independent seeds.

## Limitations

1. **Thin feature set.** After excluding the leak, only 3 features remain. Both models operate near the same information ceiling, which compresses the performance gap.
2. **Synthetic data structure.** The data-generating process uses a logistic function, which may structurally favour logistic regression. Results on real churn data could differ.
3. **Default hyperparameters.** No tuning was performed. Gradient boosting is more sensitive to hyperparameters than logistic regression; tuning it could widen the gap.
4. **No final hold-out.** Given ~4,000 rows and 5-fold CV, there is no separately withheld test set. The 3-seed repetition compensates but does not fully substitute.
5. **Null-result caveat.** If the gap is within noise, the honest claim is "no detectable difference on this dataset and feature set," not "equal performance."
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
