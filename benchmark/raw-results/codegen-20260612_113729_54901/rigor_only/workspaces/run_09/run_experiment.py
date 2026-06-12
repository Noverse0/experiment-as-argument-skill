"""Entrypoint: compare LogisticRegression vs GradientBoosting on churn data.

Run:
    python3 run_experiment.py

Outputs:
    results/metrics.json  — machine-readable results
    REPORT.md             — human-readable comparison and methodology
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data import load_and_clean, temporal_split
from src.evaluate import compute_metrics, run_seeds
from src.features import get_Xy
from src.models import make_pipeline

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
SEEDS = [0, 1, 2, 3, 4]


def sanity_checks(X_train, y_train, X_test, y_test) -> list[str]:
    """Run cheap sanity checks before full training. Returns list of check summaries."""
    checks = []

    # Baseline floor
    majority_rate = float(max(y_test.mean(), 1 - y_test.mean()))
    checks.append(f"majority_class_rate={majority_rate:.3f} (baseline AUC=0.500)")

    # Overfit one tiny subset — if GBM can't overfit 20 rows, pipeline is broken
    from sklearn.ensemble import GradientBoostingClassifier

    X_tiny, y_tiny = X_train[:20], y_train[:20]
    scaler = StandardScaler()
    X_tiny_s = scaler.fit_transform(X_tiny)
    tiny_model = GradientBoostingClassifier(n_estimators=50, random_state=0)
    tiny_model.fit(X_tiny_s, y_tiny)
    tiny_auc = roc_auc_score(y_tiny, tiny_model.predict_proba(X_tiny_s)[:, 1])
    assert tiny_auc > 0.9, f"Overfit check failed: AUC={tiny_auc:.3f} on 20-row subset"
    checks.append(f"overfit_check_auc={tiny_auc:.3f} (must be >0.9) PASS")

    # Label-shuffle: model trained on shuffled labels must score near 0.5
    rng = np.random.default_rng(99)
    y_shuffled = rng.permutation(y_train)
    scaler2 = StandardScaler()
    X_tr_s = scaler2.fit_transform(X_train)
    X_te_s = scaler2.transform(X_test)
    shuffle_model = GradientBoostingClassifier(n_estimators=50, random_state=0)
    shuffle_model.fit(X_tr_s, y_shuffled)
    shuffle_auc = roc_auc_score(y_test, shuffle_model.predict_proba(X_te_s)[:, 1])
    assert shuffle_auc < 0.6, (
        f"Label-shuffle test failed: AUC={shuffle_auc:.3f} after shuffling labels — "
        "information may be leaking around labels"
    )
    checks.append(f"label_shuffle_auc={shuffle_auc:.3f} (must be <0.6) PASS")

    return checks


def write_report(output: dict) -> None:
    lr = output["models"]["logistic"]
    gbm = output["models"]["gbm"]

    lr_auc = lr["roc_auc_mean"]
    lr_std = lr["roc_auc_std"]
    gbm_auc = gbm["roc_auc_mean"]
    gbm_std = gbm["roc_auc_std"]
    gap = gbm_auc - lr_auc

    # Conservative noise bound: gap must exceed 2× the larger std to claim a winner
    noise_bound = 2 * max(lr_std, gbm_std) if max(lr_std, gbm_std) > 0 else 0.005
    detectable = abs(gap) >= noise_bound

    if not detectable:
        conclusion = "no detectable difference between the two models"
        winner_line = "**Neither model is clearly better** — the gap is within noise."
    elif gap > 0:
        conclusion = "GradientBoostingClassifier outperforms LogisticRegression"
        winner_line = f"**GradientBoostingClassifier wins** (gap = {gap:+.4f} AUC, noise bound ≈ ±{noise_bound:.4f})."
    else:
        conclusion = "LogisticRegression outperforms GradientBoostingClassifier"
        winner_line = f"**LogisticRegression wins** (gap = {gap:+.4f} AUC, noise bound ≈ ±{noise_bound:.4f})."

    data = output["data"]
    seeds = output["seeds"]
    checks_str = "\n".join(f"- {c}" for c in output["sanity_checks"])

    report = f"""# Churn Prediction: Logistic Regression vs Gradient Boosting

## Claim

{conclusion.capitalize()} on this customer churn dataset, as measured by
ROC-AUC on a temporal hold-out test set ({len(seeds)} seeds).

## Methodology

### Leakage Prevention

| Action | Reason |
|--------|--------|
| Dropped `account_status` | Derived directly from target (`"closed"` iff `churned=1`). Perfect leak. |
| Dropped `customer_id` | Identifier; no predictive signal. |
| Deduplicated {output["audit"]["duplicates_removed"]} rows before splitting | Duplicates straddling train/test inflate test metrics. |
| Time-based split on `signup_date` | Random splits on temporal data allow future-customer rows into train. |

### Split Strategy

Records sorted by `signup_date` (converted to `signup_days` since 2023-01-01).
The first 80 % form the training set; the last 20 % form the held-out test set.

| Set | Rows | Churn rate |
|-----|------|------------|
| Train | {data["train_rows"]} | {data["train_churn_rate"]:.3f} |
| Test | {data["test_rows"]} | {data["test_churn_rate"]:.3f} |

### Features

`tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`

All features are scaled with `StandardScaler` fitted on train data only
(via `sklearn.Pipeline`, which prevents test leakage by construction).

### Models

| Model | Key hyperparameters |
|-------|---------------------|
| `LogisticRegression` | `max_iter=1000`, `solver=lbfgs` |
| `GradientBoostingClassifier` | `n_estimators=100`, `max_depth=3`, `lr=0.1`, `subsample=0.8` |

### Evaluation

Primary metric: **ROC-AUC** — robust to class imbalance and threshold-free.

Each model trained {len(seeds)} times (seeds = {seeds}) on the same fixed temporal
split. Variance across seeds captures model initialization randomness (relevant
for GBM's stochastic subsampling; LR with `lbfgs` is deterministic so its std ≈ 0).

### Sanity Checks (Passed)

{checks_str}

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| LogisticRegression | {lr_auc:.4f} ± {lr_std:.4f} | {lr["f1_mean"]:.4f} ± {lr["f1_std"]:.4f} | {lr["precision_mean"]:.4f} ± {lr["precision_std"]:.4f} | {lr["recall_mean"]:.4f} ± {lr["recall_std"]:.4f} |
| GradientBoosting | {gbm_auc:.4f} ± {gbm_std:.4f} | {gbm["f1_mean"]:.4f} ± {gbm["f1_std"]:.4f} | {gbm["precision_mean"]:.4f} ± {gbm["precision_std"]:.4f} | {gbm["recall_mean"]:.4f} ± {gbm["recall_std"]:.4f} |

Gap (GBM − LR): **{gap:+.4f}** AUC points
Noise bound (2 × max std): ±{noise_bound:.4f}

## Conclusion

{winner_line}

## Limitations

1. **Synthetic dataset**: The generative model is a logistic function of
   `tenure_months`, `monthly_spend`, and `support_tickets` — structurally
   linear. This advantages LogisticRegression; on real-world nonlinear churn
   data, GBM would likely widen its lead.
2. **No hyperparameter tuning**: GBM defaults were chosen by convention, not
   cross-validated. A tuned GBM may perform differently.
3. **Seed variance only**: The ±std reflects model initialization variance, not
   test-set sampling uncertainty. Bootstrap CIs would give a fuller picture.
4. **Temporal split caveats**: The test cohort signed up later, not at a truly
   future time. A real forward evaluation would require a later data window.
5. **LogisticRegression std ≈ 0**: `lbfgs` is deterministic; all five seeds
   produce identical results, which is honest — LR has no initialization variance.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


def main() -> None:
    # 1. Generate dataset
    print("=== Generating dataset ===")
    subprocess.run(
        [sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True
    )

    # 2. Load and clean
    print("\n=== Loading and cleaning ===")
    df, audit = load_and_clean(DATA_PATH)
    print(f"  Dropped: {audit['dropped_cols']}")
    print(f"  Duplicates removed: {audit['duplicates_removed']}")
    print(f"  Shape after cleaning: {df.shape}")

    # 3. Temporal split
    train, test = temporal_split(df, test_frac=0.2)
    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"  Train churn rate: {train['churned'].mean():.3f}")
    print(f"  Test  churn rate: {test['churned'].mean():.3f}")

    X_train, y_train = get_Xy(train)
    X_test, y_test = get_Xy(test)

    # 4. Sanity checks
    print("\n=== Sanity checks ===")
    checks = sanity_checks(X_train, y_train, X_test, y_test)
    for c in checks:
        print(f"  {c}")

    # 5. Run experiments
    print(f"\n=== Training models (seeds={SEEDS}) ===")
    model_configs = {
        "logistic": lambda seed: make_pipeline("logistic", seed),
        "gbm": lambda seed: make_pipeline("gbm", seed),
    }

    model_results = {}
    for name, pipeline_fn in model_configs.items():
        print(f"  [{name}] training...")
        model_results[name] = run_seeds(
            pipeline_fn, X_train, y_train, X_test, y_test, SEEDS
        )
        r = model_results[name]
        print(
            f"  [{name}] ROC-AUC: {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}"
        )

    # 6. Write results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "experiment": "churn_lr_vs_gbm",
        "audit": audit,
        "data": {
            "total_rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_churn_rate": float(train["churned"].mean()),
            "test_churn_rate": float(test["churned"].mean()),
        },
        "seeds": SEEDS,
        "sanity_checks": checks,
        "models": model_results,
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {metrics_path}")

    write_report(output)
    print("Report written to REPORT.md")

    # 7. Print summary
    lr = model_results["logistic"]
    gbm = model_results["gbm"]
    gap = gbm["roc_auc_mean"] - lr["roc_auc_mean"]
    print(f"\n=== Summary ===")
    print(f"  LR  ROC-AUC: {lr['roc_auc_mean']:.4f} ± {lr['roc_auc_std']:.4f}")
    print(f"  GBM ROC-AUC: {gbm['roc_auc_mean']:.4f} ± {gbm['roc_auc_std']:.4f}")
    print(f"  Gap (GBM - LR): {gap:+.4f}")


if __name__ == "__main__":
    main()
