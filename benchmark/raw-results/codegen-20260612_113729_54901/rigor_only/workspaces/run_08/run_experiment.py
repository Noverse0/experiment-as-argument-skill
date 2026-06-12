"""
Entrypoint: compare LogisticRegression vs GradientBoostingClassifier for churn prediction.

Methodology:
- Drop account_status (label leak) and deduplicate before splitting
- Temporal split: first 80% by signup_date → train, last 20% → test
- 5-fold stratified CV on training data for model comparison (mean ± std)
- Test set touched exactly once for final evaluation
- Sanity checks run before comparison (overfit tiny subset, label-shuffle)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.checks import run_sanity_checks
from src.data import load_and_clean, temporal_split
from src.evaluate import cv_evaluate
from src.pipeline import make_gb_pipeline, make_lr_pipeline

DATASET_PATH = "churn.csv"


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    # ---- 1. Load and clean ----
    print("Loading data...")
    X, y, meta = load_and_clean(DATASET_PATH)
    print(f"  {meta['n_rows']} rows after removing {meta['n_dupes_removed']} duplicates")
    print(f"  Churn rate: {meta['churn_rate']:.2%}")
    print(f"  Features: {meta['features']}")

    # ---- 2. Temporal holdout split ----
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_frac=0.20)
    print(f"\nSplit: train={len(X_train)}, test={len(X_test)}")
    print(f"  Train churn rate: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # ---- 3. Sanity checks (on LR as representative pipeline) ----
    print("\nSanity checks (LogisticRegression)...")
    checks = run_sanity_checks(make_lr_pipeline(), X_train, y_train)
    print(f"  Overfit tiny subset ROC-AUC : {checks['overfit_tiny_roc_auc']:.4f}"
          f"  [{'PASS' if checks['overfit_check_passed'] else 'FAIL'}]")
    print(f"  Label-shuffle ROC-AUC       : {checks['label_shuffle_roc_auc']:.4f}"
          f"  [{'PASS' if checks['label_shuffle_check_passed'] else 'FAIL'}]")

    if not checks["overfit_check_passed"]:
        print("  WARNING: pipeline cannot fit small data — check feature/target alignment")
    if not checks["label_shuffle_check_passed"]:
        print("  WARNING: label-shuffle test failed — suspected residual leakage")

    # ---- 4. Cross-validation comparison on training data ----
    print("\nCross-validation (5-fold stratified, training data)...")
    models = {
        "LogisticRegression": make_lr_pipeline(),
        "GradientBoosting": make_gb_pipeline(),
    }

    cv_results: dict = {}
    for name, pipeline in models.items():
        print(f"  Evaluating {name}...", flush=True)
        cv_results[name] = cv_evaluate(pipeline, X_train, y_train)
        auc = cv_results[name]["roc_auc"]
        print(f"    ROC-AUC: {auc['mean']:.4f} ± {auc['std']:.4f} (n={auc['n_folds']} folds)")

    # ---- 5. Final test evaluation (test set touched once) ----
    print("\nFinal test evaluation (held-out set)...")
    test_results: dict = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        test_results[name] = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "avg_precision": float(average_precision_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        r = test_results[name]
        print(f"  {name}: ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['avg_precision']:.4f}  F1={r['f1']:.4f}")

    # ---- 6. Persist machine-readable results ----
    output = {
        "dataset": meta,
        "split": {
            "method": "temporal (chronological by signup_date)",
            "test_frac": 0.20,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "sanity_checks": checks,
        "cv_results": cv_results,
        "test_results": test_results,
    }

    with open("results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nWrote results/metrics.json")

    # ---- 7. Write REPORT.md ----
    _write_report(output)
    print("Wrote REPORT.md")


def _write_report(results: dict) -> None:
    meta = results["dataset"]
    split = results["split"]
    cv = results["cv_results"]
    test = results["test_results"]
    checks = results["sanity_checks"]

    lr_cv = cv["LogisticRegression"]["roc_auc"]
    gb_cv = cv["GradientBoosting"]["roc_auc"]
    gap = gb_cv["mean"] - lr_cv["mean"]
    pooled_noise = (lr_cv["std"] + gb_cv["std"]) / 2

    if abs(gap) < pooled_noise:
        conclusion = (
            f"**No detectable difference.** The ROC-AUC gap is {abs(gap):.4f}, "
            f"which is within the CV noise ({pooled_noise:.4f} pooled std). "
            "Neither model is a clear winner on this dataset."
        )
    elif gap > 0:
        conclusion = (
            f"**GradientBoosting outperforms LogisticRegression** by {gap:.4f} ROC-AUC "
            f"({gb_cv['mean']:.4f} vs {lr_cv['mean']:.4f}), "
            f"a gap that exceeds the pooled CV noise ({pooled_noise:.4f})."
        )
    else:
        conclusion = (
            f"**LogisticRegression matches or outperforms GradientBoosting** "
            f"by {abs(gap):.4f} ROC-AUC ({lr_cv['mean']:.4f} vs {gb_cv['mean']:.4f}), "
            f"a gap that exceeds the pooled CV noise ({pooled_noise:.4f})."
        )

    def _fmt(cv_dict: dict, metric: str) -> str:
        d = cv_dict[metric]
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    report = f"""# Churn Prediction Experiment: GradientBoosting vs LogisticRegression

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting
customer churn on this dataset?

## Conclusion

{conclusion}

## Methodology

### Leakage Decisions
| Feature | Action | Reason |
|---------|--------|--------|
| `account_status` | **Dropped** | Derived directly from target: `"closed"` iff `churned==1`. Perfect leak. |
| `customer_id` | Dropped | Identifier; no predictive signal. |
| `signup_date` | Converted to year/month/day_of_year | Raw date dropped after numeric extraction. |

### Deduplication
{meta['n_dupes_removed']} exact duplicate rows removed **before** any split
to prevent the same row appearing in both train and test.

### Split Strategy
The dataset has a temporal column (`signup_date`). A random split would
allow future-signed customers to inform predictions about earlier ones;
a chronological split avoids this.

- **Method**: sort by `signup_date`, take last 20% as test
- Train: {split['train_size']} rows (churn rate {split['train_churn_rate']:.2%})
- Test: {split['test_size']} rows (churn rate {split['test_churn_rate']:.2%})
- Test set touched **exactly once** (final evaluation only)

### Evaluation Methodology
- **CV**: 5-fold StratifiedKFold on training data (≥3 folds required for variance)
- **Metrics**:
  - ROC-AUC — threshold-independent ranking quality
  - PR-AUC (average precision) — better for imbalanced targets
  - F1 — harmonic mean at default threshold
- Preprocessing (StandardScaler for LR) fitted inside each fold to prevent leakage

### Sanity Checks
| Check | Value | Pass? |
|-------|-------|-------|
| Overfit tiny subset (64 rows) train AUC | {checks['overfit_tiny_roc_auc']:.4f} | {'✓' if checks['overfit_check_passed'] else '✗'} |
| Label-shuffle AUC (expect ≈ 0.5) | {checks['label_shuffle_roc_auc']:.4f} | {'✓' if checks['label_shuffle_check_passed'] else '✗'} |

## Results

### Cross-Validation (Training Data, 5 Folds)
| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| LogisticRegression | {_fmt(cv['LogisticRegression'], 'roc_auc')} | {_fmt(cv['LogisticRegression'], 'avg_precision')} | {_fmt(cv['LogisticRegression'], 'f1')} |
| GradientBoosting   | {_fmt(cv['GradientBoosting'], 'roc_auc')} | {_fmt(cv['GradientBoosting'], 'avg_precision')} | {_fmt(cv['GradientBoosting'], 'f1')} |

### Final Held-Out Test Set (Touched Once)
| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| LogisticRegression | {test['LogisticRegression']['roc_auc']:.4f} | {test['LogisticRegression']['avg_precision']:.4f} | {test['LogisticRegression']['f1']:.4f} |
| GradientBoosting   | {test['GradientBoosting']['roc_auc']:.4f} | {test['GradientBoosting']['avg_precision']:.4f} | {test['GradientBoosting']['f1']:.4f} |

## Limitations
1. **Single dataset**: results may not generalize to other churn datasets with
   different feature distributions or label mechanisms.
2. **No hyperparameter tuning**: both models use moderate defaults. A tuned GB
   might show a different advantage; tuning LR's regularization strength could
   also shift results.
3. **Temporal drift**: the test window covers later signups. If customer
   behaviour shifts over time, model performance may degrade at deployment.
4. **Feature engineering**: temporal proxies (year, month, day_of_year) are
   weak representations; richer date features or interaction terms could help.
5. **Data generating process**: the true relationship is linear in log-odds
   (simulated logistic), which inherently favours LR. Real datasets may have
   non-linear interactions where GB would gain more.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
