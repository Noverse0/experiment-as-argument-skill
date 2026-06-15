"""Entrypoint: compare LogisticRegression vs GradientBoosting for churn prediction."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data_loader import get_X_y, load_and_clean
from src.evaluation import evaluate_model, sanity_checks
from src.pipeline import MODELS


DATASET_PATH = "churn.csv"
RESULTS_DIR = Path("results")
SEEDS = [42, 123, 7]
N_SPLITS = 5


def generate_dataset() -> None:
    if not Path(DATASET_PATH).exists():
        print("Generating dataset...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", DATASET_PATH],
            check=True,
        )


def write_report(
    data_stats: dict,
    model_results: dict[str, dict],
    sanity: dict[str, dict],
    outpath: Path,
) -> None:
    lr = model_results["LogisticRegression"]
    gb = model_results["GradientBoosting"]

    lr_auc = lr["auc_mean"]
    gb_auc = gb["auc_mean"]
    diff = gb_auc - lr_auc

    # Overlapping ± 2 SD intervals → no detectable difference.
    lr_hi = lr_auc + 2 * lr["auc_std"]
    gb_lo = gb_auc - 2 * gb["auc_std"]
    lr_lo = lr_auc - 2 * lr["auc_std"]
    gb_hi = gb_auc + 2 * gb["auc_std"]
    intervals_overlap = lr_hi > gb_lo and gb_hi > lr_lo

    if intervals_overlap:
        conclusion = (
            "No detectable difference: the ±2 SD AUC intervals overlap. "
            "Neither model is clearly superior on this dataset."
        )
    elif gb_auc > lr_auc:
        conclusion = (
            f"GradientBoosting outperforms LogisticRegression "
            f"(ΔAUC = {diff:+.4f}, non-overlapping ±2 SD intervals)."
        )
    else:
        conclusion = (
            f"LogisticRegression outperforms GradientBoosting "
            f"(ΔAUC = {diff:+.4f}, non-overlapping ±2 SD intervals)."
        )

    lines = [
        "# Churn Prediction Experiment Report",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Methodology",
        "",
        "### Dataset",
        f"- Raw rows: {data_stats['n_raw']}",
        f"- Duplicate rows removed: {data_stats['n_deduped']}",
        f"- Rows after dedup: {data_stats['n_clean']}",
        f"- Churn rate: {data_stats['churn_rate']:.1%}",
        "",
        "### Features Used",
        "",
        "| Feature | Notes |",
        "|---------|-------|",
        "| tenure_months | kept — causal predictor |",
        "| monthly_spend | kept — causal predictor |",
        "| support_tickets | kept — causal predictor |",
        "| signup_days | kept — days since earliest signup (derived from signup_date) |",
        "| days_since_last_login | **EXCLUDED — target leak**: churned customers stop logging in by definition; this feature is recorded after the outcome |",
        "| customer_id | excluded — identifier only |",
        "",
        "### Split Strategy",
        "",
        "TimeSeriesSplit (5 folds) on rows sorted by `signup_date`. This ensures each",
        "fold trains on customers who signed up earlier and tests on later cohorts,",
        "matching real deployment: a model trained on historical customers predicts",
        "churn for newly acquired ones. Random splits were rejected because duplicate",
        "rows could straddle and because temporal autocorrelation inflates held-out",
        "metrics.",
        "",
        "Preprocessing (StandardScaler) is fitted on the training portion of each fold",
        "only and applied to the test portion — no leakage through normalization stats.",
        "",
        f"Evaluation was repeated over {len(SEEDS)} random seeds (model internal",
        f"randomness) × {N_SPLITS} folds = {len(SEEDS) * N_SPLITS} observations per arm.",
        "",
        "### Metrics",
        "",
        "Primary: ROC-AUC (threshold-free, handles class imbalance).",
        "Secondary: F1 at default 0.5 threshold.",
        "",
        "## Results",
        "",
        "| Model | AUC mean ± SD | F1 mean ± SD | N |",
        "|-------|--------------|-------------|---|",
        f"| LogisticRegression | {lr['auc_mean']:.4f} ± {lr['auc_std']:.4f} | {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f} | {lr['n_observations']} |",
        f"| GradientBoosting   | {gb['auc_mean']:.4f} ± {gb['auc_std']:.4f} | {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f} | {gb['n_observations']} |",
        "",
        f"AUC difference (GB − LR): {diff:+.4f}",
        "",
        "## Sanity Checks",
        "",
        "| Check | LR | GB |",
        "|-------|----|----|",
        f"| Train AUC (full fit, overfit check) | {sanity['LogisticRegression']['train_auc_full']:.4f} | {sanity['GradientBoosting']['train_auc_full']:.4f} |",
        f"| Label-shuffle AUC (should be ~0.5) | {sanity['LogisticRegression']['shuffled_label_auc']:.4f} | {sanity['GradientBoosting']['shuffled_label_auc']:.4f} |",
        f"| Label-shuffle check passed | {sanity['LogisticRegression']['label_shuffle_ok']} | {sanity['GradientBoosting']['label_shuffle_ok']} |",
        "",
        "## Limitations",
        "",
        "1. **Single dataset / no held-out test set**: All evaluation is via CV; there",
        "   is no final held-out test. The test set was never touched for any decision.",
        "2. **No hyperparameter tuning**: Both models use default/fixed hyperparameters.",
        "   Tuning either arm could shift results.",
        "3. **Temporal validity**: The dataset is synthetic; real churn data may have",
        "   more complex temporal dependencies (concept drift, seasonality).",
        "4. **Feature engineering**: Only the provided columns were used. Domain-derived",
        "   features (e.g., spend trajectory, ticket velocity) could benefit tree models",
        "   more than LR.",
        "5. **`days_since_last_login` excluded**: This strong signal was removed as a",
        "   target leak. If a deployment can guarantee this feature is measured before",
        "   the churn label is recorded, the experiment should be re-run with it.",
    ]

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    generate_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and cleaning data...")
    df, data_stats = load_and_clean(DATASET_PATH)
    X, y = get_X_y(df)
    print(
        f"  {data_stats['n_clean']} rows after dedup "
        f"(removed {data_stats['n_deduped']}), "
        f"churn rate {data_stats['churn_rate']:.1%}"
    )
    print(f"  Features: {data_stats['feature_cols']}")

    model_results: dict[str, dict] = {}
    sanity: dict[str, dict] = {}

    for name, make_fn in MODELS.items():
        print(f"\nRunning {name}...")

        sanity[name] = sanity_checks(X, y, make_fn)
        print(
            f"  Sanity — train AUC: {sanity[name]['train_auc_full']:.4f}, "
            f"shuffle AUC: {sanity[name]['shuffled_label_auc']:.4f} "
            f"({'OK' if sanity[name]['label_shuffle_ok'] else 'WARN'})"
        )

        model_results[name] = evaluate_model(
            X, y, make_fn, n_splits=N_SPLITS, seeds=SEEDS
        )
        r = model_results[name]
        print(
            f"  CV AUC: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
            f"F1: {r['f1_mean']:.4f} ± {r['f1_std']:.4f}  "
            f"(n={r['n_observations']})"
        )

    # Write machine-readable results.
    output = {
        "data_stats": data_stats,
        "model_results": model_results,
        "sanity_checks": sanity,
    }
    results_path = RESULTS_DIR / "metrics.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics written to {results_path}")

    # Write report.
    report_path = Path("REPORT.md")
    write_report(data_stats, model_results, sanity, report_path)
    print(f"Report written to {report_path}")

    # Print conclusion.
    lr_auc = model_results["LogisticRegression"]["auc_mean"]
    gb_auc = model_results["GradientBoosting"]["auc_mean"]
    print(f"\n--- CONCLUSION ---")
    print(f"LR  AUC: {lr_auc:.4f} ± {model_results['LogisticRegression']['auc_std']:.4f}")
    print(f"GB  AUC: {gb_auc:.4f} ± {model_results['GradientBoosting']['auc_std']:.4f}")
    print(f"ΔAUC (GB-LR): {gb_auc - lr_auc:+.4f}")


if __name__ == "__main__":
    main()
