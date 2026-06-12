"""Entrypoint: run the full churn model comparison and write artifacts.

Usage:
    python3 make_dataset.py --out churn.csv   # produce the data first
    python3 run_experiment.py                 # writes results/metrics.json + REPORT.md

Everything needed to reproduce a run -- config, seeds, data hash, code version,
sanity outcomes, and per-fold metrics -- is recorded to results/metrics.json,
not left in console scrollback.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import sklearn

from src import data as data_mod
from src import experiment as exp

DATA_PATH = "churn.csv"
RESULTS_DIR = Path("results")
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORT_PATH = Path("REPORT.md")


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    clean = data_mod.load_clean(DATA_PATH)

    # --- sanity checks (run and recorded BEFORE trusting the comparison) ---
    X_leak, y_leak = data_mod.load_with_leak(DATA_PATH)
    sanity = {
        "majority_baseline": exp.sanity_majority_baseline(clean.X, clean.y),
        "label_shuffle": exp.sanity_label_shuffle(clean.X, clean.y),
        "overfit_tiny": exp.sanity_overfit_tiny(clean.X, clean.y),
        "leakage_ceiling": exp.sanity_leakage_ceiling(X_leak, y_leak),
    }

    # --- the comparison ---
    models = exp.make_models(exp.SEED)
    arms = exp.evaluate_arms(clean.X, clean.y, models, n_splits=exp.N_SPLITS)
    lr, gb = arms["logistic_regression"], arms["gradient_boosting"]
    delta = exp.paired_delta(lr, gb, metric="roc_auc")

    metrics = {
        "config": {
            "seed": exp.SEED,
            "n_splits": exp.N_SPLITS,
            "split": "TimeSeriesSplit on signup_date order (train=past, test=future)",
            "features": data_mod.FEATURES,
            "dropped_columns": data_mod.DROPPED,
            "preprocessing": "StandardScaler fit on train fold only (in Pipeline)",
            "primary_metric": "roc_auc",
        },
        "provenance": {
            "data_command": "python3 make_dataset.py --out churn.csv",
            "data_sha256_16": _sha256(DATA_PATH),
            "code_git_rev": _git_rev(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
        "data": {
            "n_raw": clean.n_raw,
            "n_duplicates_removed": clean.n_duplicates,
            "n_clean": clean.n_clean,
            "churn_rate": round(clean.churn_rate, 4),
        },
        "sanity": sanity,
        "arms": {
            "logistic_regression": {
                "summary": lr.summary(),
                "per_fold": lr.per_fold,
            },
            "gradient_boosting": {
                "summary": gb.summary(),
                "per_fold": gb.per_fold,
            },
        },
        "comparison": delta,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    REPORT_PATH.write_text(render_report(metrics))
    print(f"wrote {METRICS_PATH} and {REPORT_PATH}")
    return metrics


def _verdict(metrics: dict) -> str:
    d = metrics["comparison"]
    lr = metrics["arms"]["logistic_regression"]["summary"]["roc_auc"]
    gb = metrics["arms"]["gradient_boosting"]["summary"]["roc_auc"]
    # Honest claim: significant only if the paired test clears 0.05 AND the
    # spreads are informative. Otherwise "no detectable difference".
    significant = d["p_value"] < 0.05 and d["n"] >= 3
    if not significant:
        return (
            "**No detectable difference.** Across {n} time-ordered folds, gradient "
            "boosting's ROC-AUC differs from logistic regression by {md:+.4f} "
            "(sd {sd:.4f}, paired t-test p={p:.3f}). The spreads overlap and the "
            "difference is within noise, so we do not claim a winner."
        ).format(n=d["n"], md=d["mean_delta"], sd=d["sd_delta"], p=d["p_value"])
    winner = "gradient boosting" if d["mean_delta"] > 0 else "logistic regression"
    return (
        "**{w} is better** on ROC-AUC: mean paired delta {md:+.4f} "
        "(sd {sd:.4f}, n={n}, paired t-test p={p:.3f}). "
        "LR={lrm:.4f}±{lrs:.4f}, GB={gbm:.4f}±{gbs:.4f}."
    ).format(
        w=winner.capitalize(),
        md=d["mean_delta"],
        sd=d["sd_delta"],
        n=d["n"],
        p=d["p_value"],
        lrm=lr["mean"],
        lrs=lr["sd"],
        gbm=gb["mean"],
        gbs=gb["sd"],
    )


def render_report(m: dict) -> str:
    lr = m["arms"]["logistic_regression"]["summary"]
    gb = m["arms"]["gradient_boosting"]["summary"]
    s = m["sanity"]
    cfg = m["config"]
    dat = m["data"]
    rows = [
        "# Churn: Gradient Boosting vs Logistic Regression",
        "",
        "## Claim under test",
        "",
        "Does `GradientBoostingClassifier` outperform `LogisticRegression` at "
        "predicting `churned` on this dataset?",
        "",
        "## Conclusion",
        "",
        _verdict(m),
        "",
        "## Methodology",
        "",
        f"- **Single variable:** the classifier. Both arms share identical folds, "
        f"features, preprocessing, and seed ({cfg['seed']}).",
        f"- **Features used:** {', '.join(cfg['features'])}.",
        "- **Columns dropped and why:**",
    ]
    for col, why in cfg["dropped_columns"].items():
        rows.append(f"  - `{col}` — {why}")
    rows += [
        f"- **Duplicates:** {dat['n_duplicates_removed']} exact duplicate rows removed "
        f"before splitting (raw {dat['n_raw']} → clean {dat['n_clean']}); leaving them "
        "would let identical rows straddle train/test.",
        f"- **Split:** {cfg['split']}, {cfg['n_splits']} folds. Churn is forward-looking, "
        "so a random split would train on the future — we split by time instead.",
        f"- **Preprocessing:** {cfg['preprocessing']} — no fit-like step ever sees test rows.",
        f"- **Metrics:** ROC-AUC (primary) and PR-AUC, both threshold-free and robust to "
        f"the {dat['churn_rate']:.0%} positive rate. Accuracy alone would be misleading.",
        f"- **Variance:** {cfg['n_splits']} paired folds per arm; we report mean ± sd and a "
        "paired t-test on per-fold ROC-AUC, not a single number.",
        "",
        "## Sanity checks (run before trusting the comparison)",
        "",
        f"- **Majority baseline** ROC-AUC = {s['majority_baseline']['roc_auc_mean']:.3f} "
        "(expected ≈ 0.5) — models must beat this.",
        f"- **Label shuffle** ROC-AUC = {s['label_shuffle']['roc_auc_mean']:.3f} "
        "(expected ≈ 0.5) — confirms no information leaks around the labels.",
        f"- **Overfit tiny slice** train ROC-AUC = {s['overfit_tiny']['train_roc_auc']:.3f} "
        "(expected high) — the pipeline can actually learn.",
        f"- **Leakage ceiling** (with `account_status` added back) ROC-AUC = "
        f"{s['leakage_ceiling']['roc_auc_mean']:.3f} (≈ 1.0) — direct evidence that "
        "`account_status` is a target leak, justifying its removal.",
        "",
        "## Results",
        "",
        "| Arm | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | n folds |",
        "|---|---|---|---|",
        f"| Logistic regression | {lr['roc_auc']['mean']:.4f} ± {lr['roc_auc']['sd']:.4f} | "
        f"{lr['pr_auc']['mean']:.4f} ± {lr['pr_auc']['sd']:.4f} | {lr['roc_auc']['n']} |",
        f"| Gradient boosting | {gb['roc_auc']['mean']:.4f} ± {gb['roc_auc']['sd']:.4f} | "
        f"{gb['pr_auc']['mean']:.4f} ± {gb['pr_auc']['sd']:.4f} | {gb['roc_auc']['n']} |",
        "",
        f"Paired ROC-AUC delta (GB − LR): {m['comparison']['mean_delta']:+.4f} "
        f"± {m['comparison']['sd_delta']:.4f}, p = {m['comparison']['p_value']:.3f}.",
        "",
        "## Limitations",
        "",
        "- **One dataset, one generation seed.** Variance here is across time folds, "
        "not across resampled datasets; the estimate of generalization is correspondingly narrow.",
        f"- **Small fold count (n={cfg['n_splits']}).** The paired test has low power; a true "
        "small effect could be missed (Type II), so 'no detectable difference' means *not "
        "detectable at this n*, not 'provably equal'.",
        "- **Time-based folds vary in size.** Early folds train on fewer rows; AUC sd partly "
        "reflects that, not only model variability.",
        "- **Default GB hyperparameters.** Neither model was tuned; tuning budget was held at "
        "zero for both to keep the comparison fair. Results may shift under tuning.",
        "- **Synthetic, logistic-generated labels.** The data-generating process is linear in "
        "log-odds, which structurally favours logistic regression; a real churn signal with "
        "interactions could change the verdict.",
        "",
        f"_Provenance: data sha256 {m['provenance']['data_sha256_16']}, code {m['provenance']['code_git_rev']}, "
        f"sklearn {m['provenance']['sklearn_version']}, python {m['provenance']['python_version']}._",
        "",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
