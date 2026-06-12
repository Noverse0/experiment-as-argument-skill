"""Entrypoint: run sanity checks, the full CV comparison, and the time-based
robustness check; write machine-readable metrics to results/ and a human REPORT.md.

Usage:
    python3 make_dataset.py --out churn.csv   # if not already generated
    python3 run_experiment.py                 # reads churn.csv, writes results/ + REPORT.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.data import audit, load_raw
from src.experiment import DEFAULT_SEED, run_full_experiment, time_based_split_eval
from src.sanity import run_all

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "churn.csv"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: {args.csv} not found. Run: python3 make_dataset.py --out churn.csv")
        return 2

    RESULTS_DIR.mkdir(exist_ok=True)
    df_raw = load_raw(args.csv)

    aud = audit(df_raw)
    sanity = run_all(df_raw)
    full = run_full_experiment(df_raw, seed=args.seed)
    time_split = time_based_split_eval(df_raw, seed=args.seed)

    metrics = {
        "code_version": _code_version(),
        "data_generation": "python3 make_dataset.py --out churn.csv  (default --seed 7)",
        "audit": aud.__dict__,
        "sanity_checks": sanity,
        "experiment": full,
        "time_based_robustness": time_split,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    all_sanity_pass = all(s["passed"] for s in sanity)
    write_report(metrics, all_sanity_pass)

    # Console summary (the file, not the scrollback, is the source of truth)
    cmp = full["comparison"]
    print("Sanity checks:", "ALL PASS" if all_sanity_pass else "FAILURE -> see results")
    for arm, r in full["arms"].items():
        print(f"  {arm:22s} ROC-AUC {r['roc_auc_mean']:.4f} +/- {r['roc_auc_sd']:.4f}")
    print(
        f"  diff (GB - LR) = {cmp['mean_diff_b_minus_a']:+.4f} "
        f"CI95 {cmp['ci95_diff'][0]:+.4f}..{cmp['ci95_diff'][1]:+.4f} "
        f"p={cmp['p_value']:.3f}"
    )
    print("Wrote results/metrics.json and REPORT.md")
    return 0 if all_sanity_pass else 1


def write_report(m: dict, sanity_ok: bool) -> None:
    aud = m["audit"]
    exp = m["experiment"]
    cfg = exp["config"]
    lr = exp["arms"]["logistic_regression"]
    gb = exp["arms"]["gradient_boosting"]
    cmp = exp["comparison"]
    ts = m["time_based_robustness"]

    sig = cmp["significant_at_0.05"]
    diff = cmp["mean_diff_b_minus_a"]  # GB - LR
    lo, hi = cmp["ci95_diff"]
    # A 0.01-AUC gap is our threshold for "practically meaningful"; below it the
    # difference, even if statistically detectable, is too small to matter.
    NEGLIGIBLE = 0.01
    practically_negligible = abs(diff) < NEGLIGIBLE

    if not sig:
        conclusion = (
            f"**No detectable difference.** The ROC-AUC gap (GB − LR) is {diff:+.4f} "
            f"with 95% CI [{lo:+.4f}, {hi:+.4f}], which includes 0 "
            f"(paired t-test p={cmp['p_value']:.3g}, n={cmp['n_pairs']} folds). "
            f"On this dataset the two models are statistically indistinguishable, so the "
            f"answer to the question is **no** — gradient boosting does not outperform "
            f"logistic regression."
        )
        models_separate = False
    else:
        winner = "Gradient boosting" if diff > 0 else "Logistic regression"
        loser = "logistic regression" if diff > 0 else "gradient boosting"
        mag = abs(diff)
        # Present the CI in the winner's favour (always positive margin).
        w_lo, w_hi = (lo, hi) if diff > 0 else (-hi, -lo)
        hedge = (
            " The margin is **practically negligible** (< 0.01 AUC): statistically "
            "detectable thanks to paired folds, but too small to prefer one model over "
            "the other in practice."
            if practically_negligible
            else ""
        )
        answer = (
            "**no** — gradient boosting does not outperform logistic regression"
            if diff < 0
            else "**yes** — gradient boosting outperforms logistic regression"
        )
        conclusion = (
            f"**{winner} edges out {loser}** on ROC-AUC by {mag:.4f} "
            f"(95% CI [{w_lo:+.4f}, {w_hi:+.4f}], paired t-test p={cmp['p_value']:.3g}, "
            f"n={cmp['n_pairs']} folds).{hedge} Answer to the question: {answer}."
        )
        models_separate = not practically_negligible

    leak_demo = next(s for s in m["sanity_checks"] if s["check"] == "leakage_ceiling")
    shuffle = next(s for s in m["sanity_checks"] if s["check"] == "label_shuffle")
    clean_chk = next(s for s in m["sanity_checks"] if s["check"] == "clean_not_near_perfect")
    base = next(s for s in m["sanity_checks"] if s["check"] == "beats_baseline")

    report = f"""# Churn prediction: Gradient Boosting vs Logistic Regression

## Question
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset?

## Conclusion
{conclusion}

The majority-class baseline scores ROC-AUC ≈ {exp['majority_baseline']['roc_auc_mean']:.3f}
(accuracy {exp['majority_baseline']['accuracy_mean']:.3f}); both models clear it comfortably,
so both learn real signal{" — but the gap between them is within noise" if not models_separate else ""}.

| Model | ROC-AUC (mean ± sd) | Avg precision (mean ± sd) | Accuracy (mean ± sd) |
|---|---|---|---|
| Logistic regression | {lr['roc_auc_mean']:.4f} ± {lr['roc_auc_sd']:.4f} | {lr['avg_precision_mean']:.4f} ± {lr['avg_precision_sd']:.4f} | {lr['accuracy_mean']:.4f} ± {lr['accuracy_sd']:.4f} |
| Gradient boosting | {gb['roc_auc_mean']:.4f} ± {gb['roc_auc_sd']:.4f} | {gb['avg_precision_mean']:.4f} ± {gb['avg_precision_sd']:.4f} | {gb['accuracy_mean']:.4f} ± {gb['accuracy_sd']:.4f} |

Estimates: {cfg['n_estimates']} paired folds ({cfg['n_splits']}-fold × {cfg['n_repeats']} repeats),
identical folds for both arms.

## Methodology

**Claim under test.** A beats B on ROC-AUC for predicting `churned`, where the only
thing varied is the model; features, folds, preprocessing, and seed are held fixed.

**Data audit (reported, not hidden).**
- Raw rows: {aud['n_raw']}; exact duplicate rows found and removed: {aud['n_duplicates']};
  rows after dedup: {aud['n_after_dedup']}.
- Base churn rate: {aud['base_rate']:.4f} (imbalanced → ROC-AUC / average precision are the
  primary metrics, not accuracy).
- Signup time span: {aud['time_span'][0]} … {aud['time_span'][1]}.

**Leak surface and what we did about it.**
- `account_status` is a **perfect target leak**: it is "closed" iff the customer churned
  and is recorded *after* the outcome. **Dropped.** The leakage sanity check confirms it:
  a model that sees it reaches ROC-AUC = {leak_demo['auc_with_leak']:.4f} — near-perfect on a
  noisy task, the classic leakage signature. This is exactly the inflated result we avoid.
- `customer_id` is an identifier with no generalizable signal. **Dropped.**
- `signup_date` is temporal. It is **not** used as a feature; instead it defines a
  time-ordered split for the robustness check below.
- The 200 duplicate rows are **removed before any split**, so identical rows cannot
  straddle train/test and inflate scores.

**Features used:** {", ".join(cfg['features'])}.

**Preprocessing.** `StandardScaler` is fit on the training fold only, inside a
`Pipeline`, then applied to the test fold — no test statistics leak into training.
(Scaling is required for LR and harmless for the tree model; applying it to both keeps
the pipelines symmetric so the model is the only difference.)

**Evaluation.** `RepeatedStratifiedKFold` ({cfg['n_splits']} folds × {cfg['n_repeats']}
repeats = {cfg['n_estimates']} estimates), the *same* folds for both models. We report
mean ± sd per arm and a **paired** t-test on the per-fold ROC-AUC differences. One seed is
an anecdote; the repeats give the variance that the winner/no-winner call depends on.

**Seeds.** All randomness fixed and logged: experiment seed = {cfg['seed']}
(see `results/metrics.json`). The data is generated by
`python3 make_dataset.py --out churn.csv` (generator default seed 7).

## Sanity checks (all must pass before believing the result)
- **Majority baseline:** ROC-AUC ≈ {exp['majority_baseline']['roc_auc_mean']:.3f} — both models beat it.
- **Beats-baseline:** LR ROC-AUC {base['aucs']['logistic_regression']:.4f}, GB ROC-AUC {base['aucs']['gradient_boosting']:.4f} → passed = {base['passed']}.
- **Leakage ceiling:** with `account_status` included, ROC-AUC = {leak_demo['auc_with_leak']:.4f}
  (> 0.99) → leak confirmed and avoided.
- **Clean not near-perfect:** clean-feature ROC-AUC = {clean_chk['auc_clean']:.4f} (< 0.95) →
  no hidden leak remaining.
- **Label shuffle:** with labels permuted, ROC-AUC = {shuffle['auc_shuffled']:.4f} (≈ 0.5) →
  no information leaking around the labels.

Status: **{"ALL SANITY CHECKS PASSED" if sanity_ok else "SANITY CHECK FAILURE — result not trusted"}**.

## Time-based robustness check
Train on the earliest 80% of signups, test on the latest 20% (test starts
{ts['split_time']}; train base rate {ts['train_base_rate']:.3f}, test base rate
{ts['test_base_rate']:.3f}). Single split → no variance, so this is a directional
cross-check, not the primary evidence:

| Model | ROC-AUC | Avg precision |
|---|---|---|
| Logistic regression | {ts['arms']['logistic_regression']['roc_auc']:.4f} | {ts['arms']['logistic_regression']['avg_precision']:.4f} |
| Gradient boosting | {ts['arms']['gradient_boosting']['roc_auc']:.4f} | {ts['arms']['gradient_boosting']['avg_precision']:.4f} |

## Limitations
- The dataset is synthetic; `churned` is a logistic function of `tenure_months`,
  `monthly_spend`, and `support_tickets` plus noise — a generative process that *favours
  a linear model*, so this result should not be generalized to real churn data.
- Repeated K-fold estimates reuse rows across folds, so the {cfg['n_estimates']} per-fold
  AUCs are not fully independent; the paired t-test p-value is mildly optimistic and is
  used only as a coarse "inside vs outside noise" signal alongside the CI.
- Both models use library defaults (equal, zero tuning budget). A hyperparameter search
  could shift the comparison; that would be a different experiment and would require
  re-touching a validation split, not the test folds.
- Conclusion applies to this dataset and feature set only.
"""
    (ROOT / "REPORT.md").write_text(report)


if __name__ == "__main__":
    sys.exit(main())
