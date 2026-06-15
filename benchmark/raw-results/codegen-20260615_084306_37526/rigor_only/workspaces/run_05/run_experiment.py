"""Entrypoint: runs the full churn prediction experiment and writes results."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import run_cv_comparison, run_temporal_holdout, run_sanity_checks

RESULTS_DIR = Path("results")
DATA_PATH = Path("churn.csv")


def _generate_data():
    if not DATA_PATH.exists():
        print("Generating dataset...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", str(DATA_PATH)],
            check=True,
        )
    else:
        print(f"Using existing {DATA_PATH}")


def _gap_and_verdict(cv_results: dict) -> str:
    lr = cv_results["logistic_regression"]
    gb = cv_results["gradient_boosting"]
    gap = gb["roc_auc_mean"] - lr["roc_auc_mean"]
    # Pooled uncertainty: half-width of 1 SD for quick heuristic
    noise = max(lr["roc_auc_std"], gb["roc_auc_std"])
    if abs(gap) < noise:
        return "no detectable difference", gap
    elif gap > 0:
        return "gradient boosting outperforms logistic regression", gap
    else:
        return "logistic regression outperforms gradient boosting", gap


def _write_report(cv: dict, temporal: dict, sanity: dict, verdict: str, gap: float):
    lr_cv = cv["logistic_regression"]
    gb_cv = cv["gradient_boosting"]
    lr_t = temporal["logistic_regression"]
    gb_t = temporal["gradient_boosting"]

    lines = [
        "# Churn Prediction: Gradient Boosting vs Logistic Regression",
        "",
        "## Claim",
        "Does gradient boosting outperform logistic regression for predicting customer churn",
        "on the provided dataset, using only features available before the churn event?",
        "",
        "## Methodology",
        "",
        "### Data Preparation",
        "- **Deduplicated** 200 exact-duplicate rows before any split (4200 → 4000 rows).",
        "- **Dropped `days_since_last_login`**: this column is derived from the churn outcome",
        "  (churned customers stop logging in), making it a post-hoc target leak.",
        "  A model trained with it would not generalize to real deployment where the outcome",
        "  is unknown at prediction time.",
        "- **Dropped `customer_id`**: identifier with no predictive meaning.",
        "- **Engineered `days_since_signup`** from `signup_date` (days from 2023-01-01).",
        "- Final features: `tenure_months`, `monthly_spend`, `support_tickets`, `days_since_signup`.",
        "",
        "### Evaluation",
        "- **Primary**: 5-fold stratified cross-validation, repeated with 3 different seeds",
        f"  (15 fold scores per model, n={lr_cv['n_folds']}).",
        "  `StandardScaler` fitted on each training fold only (no leakage through scaling).",
        "- **Secondary**: time-ordered holdout — sort by `signup_date`, 80% train / 20% test.",
        "  This respects the temporal structure of the data.",
        "- **Metric**: ROC-AUC (primary) and F1 (supporting). ROC-AUC is preferred over accuracy",
        "  because the target rate is ~27%, making accuracy misleading.",
        "",
        "### Sanity Checks",
        f"- Majority-class baseline AUC: {sanity['majority_baseline_auc']:.3f} (floor)",
        f"- GB on legitimate features: {sanity['gb_auc_on_legitimate_features']:.3f}",
        f"- Label-shuffle AUC: {sanity['label_shuffle_auc']:.3f} (should be ≈ baseline)",
        f"- Leakage flag (AUC > 0.97): {sanity['leakage_flag']}",
        f"- Shuffle degraded as expected: {sanity['shuffle_degraded_as_expected']}",
        "",
        "## Results",
        "",
        "### Cross-Validation (n=15 folds each)",
        "",
        "| Model | ROC-AUC mean ± SD | F1 mean ± SD |",
        "|---|---|---|",
        f"| Logistic Regression | {lr_cv['roc_auc_mean']:.4f} ± {lr_cv['roc_auc_std']:.4f} | {lr_cv['f1_mean']:.4f} ± {lr_cv['f1_std']:.4f} |",
        f"| Gradient Boosting   | {gb_cv['roc_auc_mean']:.4f} ± {gb_cv['roc_auc_std']:.4f} | {gb_cv['f1_mean']:.4f} ± {gb_cv['f1_std']:.4f} |",
        "",
        "### Time-Based Holdout",
        "",
        "| Model | ROC-AUC | F1 | Accuracy |",
        "|---|---|---|---|",
        f"| Logistic Regression | {lr_t['roc_auc']:.4f} | {lr_t['f1']:.4f} | {lr_t['accuracy']:.4f} |",
        f"| Gradient Boosting   | {gb_t['roc_auc']:.4f} | {gb_t['f1']:.4f} | {gb_t['accuracy']:.4f} |",
        "",
        "## Conclusion",
        "",
        f"**Finding: {verdict}** (ROC-AUC gap = {gap:+.4f}).",
        "",
    ]

    gap_abs = abs(gap)
    noise = max(lr_cv["roc_auc_std"], gb_cv["roc_auc_std"])
    if gap_abs < noise:
        lines += [
            "The gap between models is within one standard deviation of cross-validated scores.",
            "With 15 fold evaluations, overlapping score distributions prevent a confident claim",
            "that one model is superior. The legitimate causal signal in this dataset is weak",
            "(low-magnitude logit coefficients in the generative process), and without the leaky",
            "`days_since_last_login` feature, both models are working from similarly limited signal.",
        ]
    elif gap > 0:
        lines += [
            f"Gradient boosting achieves higher ROC-AUC by {gap:.4f} on CV and by",
            f"{gb_t['roc_auc'] - lr_t['roc_auc']:+.4f} on the temporal holdout.",
            "The gap exceeds the within-model standard deviation, suggesting a real (if modest)",
            "advantage from the non-linear boosted model.",
        ]
    else:
        lines += [
            "Logistic regression performs comparably or better. On a small, clean dataset with",
            "mostly linear relationships, the added complexity of gradient boosting does not help.",
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "- **Synthetic data**: the generative process is known; real churn datasets are noisier",
        "  and contain higher-value features not present here.",
        "- **No hyperparameter tuning**: both models use default/modest settings. Proper tuning",
        "  would require an additional validation split and is outside the scope of this comparison.",
        "- **No formal significance test**: with 15 folds and correlated scores (shared data),",
        "  a proper paired test (e.g. corrected resampled t-test) was not applied. The ± SD",
        "  comparison is a heuristic, not a p-value.",
        "- **Temporal split is approximate**: the split respects signup cohort ordering but does",
        "  not guarantee a meaningful real-world train/test horizon.",
        "- **`days_since_last_login` was excluded**: if this feature were collected and logged",
        "  *before* the churn outcome is determined (e.g. as a leading indicator), it could be",
        "  legitimately used. In this dataset's construction it is post-hoc.",
    ]

    Path("REPORT.md").write_text("\n".join(lines) + "\n")
    print("Wrote REPORT.md")


def main():
    _generate_data()
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['churned'].mean():.1%} churn rate")

    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n[1/3] Running sanity checks...")
    sanity = run_sanity_checks(df)
    print(f"  Baseline AUC: {sanity['majority_baseline_auc']:.3f}")
    print(f"  GB AUC (legitimate features): {sanity['gb_auc_on_legitimate_features']:.3f}")
    print(f"  Label-shuffle AUC: {sanity['label_shuffle_auc']:.3f}")
    print(f"  Leakage flag: {sanity['leakage_flag']}")
    if sanity["leakage_flag"]:
        print("  WARNING: AUC > 0.97 detected — check for target leakage before proceeding.")
    if not sanity["shuffle_degraded_as_expected"]:
        print("  WARNING: label shuffle did not degrade performance — investigate pipeline.")

    print("\n[2/3] Running cross-validation comparison (5-fold × 3 seeds)...")
    cv_results = run_cv_comparison(df, n_folds=5, seeds=[0, 1, 2])
    for name, r in cv_results.items():
        print(f"  {name}: ROC-AUC {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}")

    print("\n[3/3] Running temporal holdout comparison...")
    temporal_results = run_temporal_holdout(df)
    for name, r in temporal_results.items():
        print(f"  {name}: ROC-AUC {r['roc_auc']:.4f}, F1 {r['f1']:.4f}")

    verdict, gap = _gap_and_verdict(cv_results)
    print(f"\nVerdict: {verdict} (gap={gap:+.4f})")

    metrics = {
        "sanity": sanity,
        "cv_comparison": cv_results,
        "temporal_holdout": temporal_results,
        "verdict": verdict,
        "roc_auc_gap_gb_minus_lr": gap,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")

    _write_report(cv_results, temporal_results, sanity, verdict, gap)
    print("Done.")


if __name__ == "__main__":
    main()
