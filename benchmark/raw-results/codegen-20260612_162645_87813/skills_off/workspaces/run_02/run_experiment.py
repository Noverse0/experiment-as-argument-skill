"""Entrypoint: run the full churn experiment and write artifacts.

Usage:
    python3 run_experiment.py [--csv churn.csv] [--n-splits 5]

Writes:
    results/metrics.json   machine-readable metrics, config, seeds, provenance
    REPORT.md              the comparison conclusion, methodology, limitations
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Make the src/ layout importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from churn.data import (  # noqa: E402
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LEAK_COLUMNS,
    load_clean,
)
from churn.experiment import N_SPLITS, run_comparison, run_sanity_checks  # noqa: E402
from churn.models import RANDOM_STATE  # noqa: E402

DATA_GEN_CMD = "python3 make_dataset.py --out {out}"


def ensure_dataset(csv_path: str) -> str:
    """Generate the dataset deterministically if it is missing. Returns the
    command used, for the provenance record."""
    cmd = DATA_GEN_CMD.format(out=csv_path)
    if not os.path.exists(csv_path):
        subprocess.run(cmd.split(), check=True)
    return cmd


def code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="churn.csv")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report", default="REPORT.md")
    args = parser.parse_args()

    data_gen_cmd = ensure_dataset(args.csv)
    data = load_clean(args.csv)

    sanity = run_sanity_checks(data, args.csv)
    comparison = run_comparison(data, n_splits=args.n_splits)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "config": {
            "random_state": RANDOM_STATE,
            "n_splits": args.n_splits,
            "split_strategy": "TimeSeriesSplit (expanding window) on signup_date",
            "features": FEATURE_COLUMNS,
            "dropped_leak_columns": LEAK_COLUMNS,
            "dropped_id_columns": ID_COLUMNS,
            "preprocessing": "StandardScaler fit on train fold only (in Pipeline)",
        },
        "data_provenance": {
            "csv_path": args.csv,
            "generation_command": data_gen_cmd,
            "n_rows_raw": data.n_raw,
            "n_duplicates_removed": data.n_duplicates_removed,
            "n_rows_used": int(len(data.X)),
            "churn_rate": data.churn_rate,
        },
        "sanity_checks": sanity,
        "results": comparison,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    write_report(args.report, metrics)

    print(f"Wrote {metrics_path} and {args.report}")
    print(f"Verdict: {comparison['comparison']['verdict']}")
    return 0


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.4f} +/- {stat['sd']:.4f}"


def write_report(path: str, m: dict) -> None:
    r = m["results"]
    c = r["comparison"]
    prov = m["data_provenance"]
    sane = m["sanity_checks"]
    gb = r["arms"]["gradient_boosting"]
    lr = r["arms"]["logistic_regression"]

    verdict_text = {
        "no_detectable_difference": (
            "**No detectable difference.** The per-fold ROC AUC gap "
            f"({c['mean_diff']:+.4f}) does not clear its own spread "
            f"(sd {c['sd_diff']:.4f}), so on this dataset and methodology we "
            "cannot say gradient boosting outperforms logistic regression."
        ),
        "gradient_boosting_better": (
            "**Gradient boosting wins.** It beats logistic regression on ROC "
            f"AUC by {c['mean_diff']:+.4f} per fold, consistently across folds "
            f"(p={c['paired_p_value']:.3f})."
        ),
        "logistic_regression_better": (
            "**Logistic regression wins.** Gradient boosting trails by "
            f"{c['mean_diff']:+.4f} ROC AUC per fold "
            f"(p={c['paired_p_value']:.3f})."
        ),
    }[c["verdict"]]

    lines = [
        "# Churn prediction: gradient boosting vs. logistic regression",
        "",
        "## Claim",
        "",
        "Does `GradientBoostingClassifier` outperform `LogisticRegression` at "
        "predicting `churned` on this dataset? The single variable is the "
        "model; preprocessing, features, splits, and tuning budget (library "
        "defaults for both) are held fixed.",
        "",
        "## Conclusion",
        "",
        verdict_text,
        "",
        f"Primary metric: ROC AUC over {r['n_splits']} forward-looking folds "
        "(mean +/- sd):",
        "",
        "| Model | ROC AUC | Avg precision (PR AUC) | Accuracy | F1 |",
        "|---|---|---|---|---|",
        f"| Gradient boosting | {_fmt(gb['roc_auc'])} | "
        f"{_fmt(gb['average_precision'])} | {_fmt(gb['accuracy'])} | "
        f"{_fmt(gb['f1'])} |",
        f"| Logistic regression | {_fmt(lr['roc_auc'])} | "
        f"{_fmt(lr['average_precision'])} | {_fmt(lr['accuracy'])} | "
        f"{_fmt(lr['f1'])} |",
        "",
        f"Paired per-fold ROC AUC difference (GB - LR): "
        f"{c['mean_diff']:+.4f} +/- {c['sd_diff']:.4f} "
        f"(n={r['n_splits']}, heuristic paired t-test p={c['paired_p_value']:.3f}).",
        "",
        "## Methodology",
        "",
        "**Leak defenses (applied before any model sees the data).** This "
        "dataset carries three hazards, all neutralized in `src/churn/data.py`:",
        "",
        "- `account_status` is **dropped**: it is `\"closed\"` exactly when the "
        "customer churned, i.e. a post-outcome copy of the target. Including it "
        "would be target leakage. The `leakage_ceiling` sanity check confirms "
        f"it yields ROC AUC {sane['leakage_ceiling']['mean_roc_auc']:.4f} on its "
        "own — proof it is the answer, not a feature.",
        f"- **{prov['n_duplicates_removed']} exact duplicate rows** are removed "
        "on the full frame before splitting, so no duplicate can straddle the "
        "train/test boundary and inflate scores.",
        "- `signup_date` is temporal and churn is forward-looking, so we use a "
        "**time-based split** (`TimeSeriesSplit`, expanding window) on "
        "signup-date order. Each test fold is strictly later than its training "
        "data. A random split would leak the future into the past.",
        "- `customer_id` is dropped as a bare identifier.",
        "",
        "**Preprocessing.** A `StandardScaler` is fit inside the pipeline on the "
        "training fold only, so no test statistics leak into fitting. Features "
        f"used: {', '.join('`%s`' % f for f in FEATURE_COLUMNS)}.",
        "",
        "**Evaluation.** "
        f"{r['n_splits']} expanding-window folds give {r['n_splits']} "
        "measurements per model, reported as mean +/- sd rather than a single "
        "split. Metrics: ROC AUC (primary; robust to the "
        f"{prov['churn_rate']:.1%} churn rate / class imbalance), average "
        "precision, plus accuracy and F1 for context. The comparison is paired "
        "per fold. Seed for all randomness: "
        f"{m['config']['random_state']}.",
        "",
        "**Sanity checks (all must pass before trusting the comparison).**",
        "",
        f"- Baseline floor: a no-information `DummyClassifier` sits at ROC AUC "
        f"{sane['baseline_floor']['mean_roc_auc']:.4f} "
        f"({'pass' if sane['baseline_floor']['passed'] else 'FAIL'}).",
        f"- Label shuffle: with labels permuted, gradient boosting collapses to "
        f"ROC AUC {sane['label_shuffle']['mean_roc_auc']:.4f} "
        f"({'pass' if sane['label_shuffle']['passed'] else 'FAIL'}) — no "
        "information leaks around the labels.",
        f"- Leakage ceiling: the dropped `account_status` alone scores "
        f"{sane['leakage_ceiling']['mean_roc_auc']:.4f} "
        f"({'pass' if sane['leakage_ceiling']['passed'] else 'FAIL'}).",
        "",
        "## Data provenance",
        "",
        f"- Generated by: `{prov['generation_command']}`",
        f"- Raw rows: {prov['n_rows_raw']}; duplicates removed: "
        f"{prov['n_duplicates_removed']}; rows used: {prov['n_rows_used']}.",
        f"- Churn rate: {prov['churn_rate']:.4f}.",
        f"- Code version: `{m['code_version']}`; run at {m['generated_at_utc']}.",
        "",
        "## Limitations",
        "",
        "- Expanding-window folds share training data, so they are not "
        "independent; the paired t-test is a heuristic, and the honest verdict "
        "leans on the per-fold spread, not the p-value.",
        "- Both models use library defaults (equal, minimal tuning budget). A "
        "different result is possible after fair, budget-matched tuning of both "
        "arms; that would be a separate experiment.",
        "- The signal in this dataset is a roughly linear function of three "
        "features, which is close to logistic regression's hypothesis class — a "
        "context where extra model capacity is not expected to help much.",
        "- Conclusions apply to this dataset and methodology only.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
