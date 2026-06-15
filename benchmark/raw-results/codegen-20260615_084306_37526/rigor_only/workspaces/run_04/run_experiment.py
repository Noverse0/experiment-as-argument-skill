#!/usr/bin/env python3
"""
Churn prediction experiment: Gradient Boosting vs Logistic Regression.

Runs sanity checks, cross-validation for both model arms, writes
results/metrics.json and REPORT.md.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_and_prepare, LEAKY_COLS
from src.evaluate import run_cv
from src.pipeline import MODELS

N_SPLITS = 5
N_REPEATS = 3


def generate_dataset() -> None:
    print("Generating dataset...")
    result = subprocess.run(
        [sys.executable, "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Dataset generation failed:\n{result.stderr}")
    print(" ", result.stdout.strip())


def _cv_scores(estimator, X, y) -> tuple[float, float]:
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores))


def audit_leaky_feature(X_audit, y) -> tuple[float, float]:
    """AUC of days_since_last_login alone — should be suspiciously high."""
    leak_X = X_audit[[LEAKY_COLS[0]]]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    return _cv_scores(pipe, leak_X, y)


def run_baseline(X, y) -> tuple[float, float]:
    return _cv_scores(DummyClassifier(strategy="most_frequent"), X, y)


def determine_conclusion(lr_mean, lr_std, gb_mean, gb_std) -> str:
    gap = gb_mean - lr_mean
    noise = (lr_std ** 2 + gb_std ** 2) ** 0.5
    if abs(gap) <= noise:
        return "no_detectable_difference"
    return "gradient_boosting_wins" if gap > 0 else "logistic_regression_wins"


def write_report(m: dict) -> None:
    lr = m["model_results"]["LogisticRegression"]
    gb = m["model_results"]["GradientBoosting"]
    sc = m["sanity_checks"]
    ds = m["data_stats"]

    conclusion_sentences = {
        "gradient_boosting_wins": (
            "**Gradient Boosting outperforms Logistic Regression** on ROC-AUC "
            "(gap exceeds combined noise floor)."
        ),
        "logistic_regression_wins": (
            "**Logistic Regression outperforms Gradient Boosting** on ROC-AUC "
            "(gap exceeds combined noise floor)."
        ),
        "no_detectable_difference": (
            "**No detectable difference** between the two models — the AUC gap "
            f"({m['gap_roc_auc']:+.3f}) is within combined noise "
            f"({m['noise_floor']:.3f})."
        ),
    }
    conclusion_text = conclusion_sentences[m["conclusion"]]

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Feature excluded (target leakage):** `days_since_last_login`
This column is post-outcome: churned customers stop logging in, so the value is
recorded *after* the churn event and is statistically derived from the label.
A single-feature LR using only this column achieves
ROC-AUC {sc['leaky_feature_auc']['mean']:.3f} ± {sc['leaky_feature_auc']['std']:.3f},
confirming it carries label information that would not be available before the
outcome in a real deployment.

**Deduplication:** {ds['n_dupes_dropped']} exact duplicate rows dropped *before*
any split. Omitting this step would let duplicates straddle the boundary and
inflate test scores. Clean dataset: {ds['n_clean']} rows.

**Evaluation:** RepeatedStratifiedKFold ({N_SPLITS} folds × {N_REPEATS} repeats =
{N_SPLITS * N_REPEATS} scores per model). Stratification preserves the
{ds['churn_rate_clean']:.1%} churn rate across folds. `StandardScaler` is fitted
*inside* each fold's training split only — no preprocessing leakage.

**Primary metric:** ROC-AUC (robust to class imbalance).
Secondary metrics: F1, accuracy.

## Sanity Checks

| Check | Result | Interpretation |
|---|---|---|
| Majority-class baseline AUC | {sc['baseline_auc']['mean']:.3f} ± {sc['baseline_auc']['std']:.3f} | Floor; both models must exceed this |
| Leaky-feature-only AUC | {sc['leaky_feature_auc']['mean']:.3f} ± {sc['leaky_feature_auc']['std']:.3f} | Confirmed post-outcome leak — excluded |

## Results

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|
| LogisticRegression | {lr['roc_auc']['mean']:.3f} ± {lr['roc_auc']['std']:.3f} | {lr['f1']['mean']:.3f} ± {lr['f1']['std']:.3f} | {lr['accuracy']['mean']:.3f} ± {lr['accuracy']['std']:.3f} |
| GradientBoosting | {gb['roc_auc']['mean']:.3f} ± {gb['roc_auc']['std']:.3f} | {gb['f1']['mean']:.3f} ± {gb['f1']['std']:.3f} | {gb['accuracy']['mean']:.3f} ± {gb['accuracy']['std']:.3f} |

Gap (GB − LR) ROC-AUC: {m['gap_roc_auc']:+.3f}
Noise floor (combined std): {m['noise_floor']:.3f}

n = {N_SPLITS * N_REPEATS} CV folds per model on {ds['n_clean']} deduplicated rows.

## Conclusion

{conclusion_text}

## Limitations

1. **No hyperparameter tuning.** Both models use defaults. Tuned variants might shift or reverse the gap.
2. **Temporal structure partially ignored.** `signup_date` was dropped; a strict time-based split (train-early / test-late) would better simulate production deployment.
3. **Small honest feature set.** After removing the leak, only 3 features remain. Additional non-leaky features could change the relative advantage of each model.
4. **Single dataset.** Results are specific to this DGP (n={ds['n_clean']}); they may not generalise to real churn datasets.
"""
    with open("REPORT.md", "w") as f:
        f.write(report)


def main() -> None:
    generate_dataset()

    print("Loading and preparing data...")
    X, y, X_audit, data_stats = load_and_prepare("churn.csv")
    print(f"  Rows (raw → clean): {data_stats['n_raw']} → {data_stats['n_clean']} "
          f"({data_stats['n_dupes_dropped']} duplicates dropped)")
    print(f"  Churn rate: {data_stats['churn_rate_clean']:.1%}")

    print("\nSanity checks...")
    leak_mean, leak_std = audit_leaky_feature(X_audit, y)
    print(f"  Leaky-feature-only AUC: {leak_mean:.3f} ± {leak_std:.3f}  "
          f"(suspiciously high → confirmed leak, excluded)")

    baseline_mean, baseline_std = run_baseline(X, y)
    print(f"  Majority-class baseline AUC: {baseline_mean:.3f} ± {baseline_std:.3f}")

    print(f"\nRunning CV ({N_SPLITS}-fold × {N_REPEATS} repeats) …")
    model_results = {}
    for name, pipeline_fn in MODELS.items():
        print(f"  {name} …", end=" ", flush=True)
        summary = run_cv(pipeline_fn, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS)
        model_results[name] = summary
        auc = summary["roc_auc"]
        print(f"ROC-AUC {auc['mean']:.3f} ± {auc['std']:.3f}")

    lr_mean = model_results["LogisticRegression"]["roc_auc"]["mean"]
    lr_std = model_results["LogisticRegression"]["roc_auc"]["std"]
    gb_mean = model_results["GradientBoosting"]["roc_auc"]["mean"]
    gb_std = model_results["GradientBoosting"]["roc_auc"]["std"]

    gap = gb_mean - lr_mean
    noise = (lr_std ** 2 + gb_std ** 2) ** 0.5
    conclusion = determine_conclusion(lr_mean, lr_std, gb_mean, gb_std)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_stats": data_stats,
        "sanity_checks": {
            "leaky_feature_auc": {"mean": leak_mean, "std": leak_std},
            "baseline_auc": {"mean": baseline_mean, "std": baseline_std},
        },
        "cv_config": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS},
        "model_results": model_results,
        "gap_roc_auc": float(gap),
        "noise_floor": float(noise),
        "conclusion": conclusion,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    write_report(output)

    print(f"\nConclusion: {conclusion}")
    print("Results written to results/metrics.json")
    print("Report written to REPORT.md")


if __name__ == "__main__":
    main()
