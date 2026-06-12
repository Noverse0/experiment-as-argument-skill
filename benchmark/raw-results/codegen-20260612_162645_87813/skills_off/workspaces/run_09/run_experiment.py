"""Entrypoint: run the full churn experiment and write results/ + REPORT.md.

Usage:
    python3 make_dataset.py --out churn.csv   # generate data (run once)
    python3 run_experiment.py                 # run experiment

Finishes in well under 5 minutes on CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from churn import data as data_mod  # noqa: E402
from churn import experiment as exp  # noqa: E402

DATA_PATH = ROOT / "churn.csv"
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "REPORT.md"


def _fmt_arm(summary: dict) -> str:
    lines = []
    for metric in ["roc_auc", "average_precision", "accuracy", "f1"]:
        s = summary[metric]
        lines.append(f"| {metric} | {s['mean']:.4f} | {s['sd']:.4f} |")
    return "\n".join(lines)


def write_report(result: dict) -> str:
    cfg, d = result["config"], result["data"]
    san = result["sanity"]
    paired = result["paired_primary"]
    lr_sum = result["summary"]["logistic_regression"]
    gb_sum = result["summary"]["gradient_boosting"]
    conclusion = exp.conclusion_text(result)

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset? Decided by ROC AUC across time-ordered CV folds.

## Conclusion

{conclusion}

## Methodology

- **Single variable:** the classifier. Features, preprocessing, splits, and seeds
  are identical across both arms, so any AUC gap is attributable to the model.
- **Evaluation:** `TimeSeriesSplit` (forward-chaining), `n_splits={cfg['n_splits']}`.
  Rows are ordered by `signup_date`; every test fold is later in signup time than
  its training data. This is a forward-looking evaluation, not a random split.
- **Preprocessing:** `{cfg['preprocessing']}` inside a `Pipeline`, so the scaler
  never sees test-fold statistics.
- **Features used:** {', '.join(f'`{c}`' for c in cfg['features'])}.
- **Seed:** {cfg['seed']} (fixed and logged; results are reproducible).
- **Metrics:** ROC AUC (primary), average precision (PR AUC, imbalance-aware),
  accuracy, and F1. Accuracy alone is not trusted given class imbalance.

### Why these columns were dropped (leakage audit, done before coding)

- **`account_status` — TARGET LEAK, dropped.** It equals `"closed"` exactly when
  `churned == 1` and `"active"` otherwise. It is a recorded-after-the-outcome proxy
  for the label. The leakage-ceiling sanity check below shows it drives AUC to
  ~1.0; including it would make the comparison meaningless.
- **`customer_id` — identifier, dropped.** No signal; risks memorization.
- **`signup_date` — temporal, used to order the split, not as a feature.** It
  carries no churn signal here but defines chronological order for the time split.

### Data hygiene

- Raw rows: {d['n_raw']}. **{d['n_duplicates_removed']} exact duplicate rows removed
  before splitting** (so identical rows cannot straddle the train/test boundary).
  Rows after dedup: {d['n_after_dedup']}.
- Base churn rate: **{d['churn_rate']:.4f}** (imbalanced — hence AUC / PR AUC).

## Results (mean +/- sd over {paired['n_folds']} folds)

### Logistic Regression
| metric | mean | sd |
|---|---|---|
{_fmt_arm(lr_sum)}

### Gradient Boosting
| metric | mean | sd |
|---|---|---|
{_fmt_arm(gb_sum)}

### Paired comparison on {paired['metric']} (GB - LR, per fold)
- Mean difference: **{paired['gb_minus_lr_mean']:+.4f} +/- {paired['gb_minus_lr_sd']:.4f}**
- Per-fold differences: {', '.join(f'{x:+.4f}' for x in paired['per_fold_diff'])}
- Paired t-test: t = {paired['paired_t_stat']:.3f}, p = {paired['paired_p_value']:.3f}
  (n={paired['n_folds']} folds — treat as a weak signal, not definitive proof)

## Sanity checks

| check | expected | observed | pass |
|---|---|---|---|
| Baseline floor (DummyClassifier AUC) | ~0.50 | {san['baseline_auc_mean']:.4f} | {'YES' if abs(san['baseline_auc_mean'] - 0.5) < 0.1 else 'NO'} |
| Label-shuffle AUC (LR) | ~0.50 | {san['label_shuffle_auc_mean']:.4f} | {'YES' if abs(san['label_shuffle_auc_mean'] - 0.5) < 0.1 else 'NO'} |
| Leakage ceiling w/ account_status | ~1.00 | {san['leakage_ceiling_auc_mean']:.4f} | {'YES' if san['leakage_ceiling_auc_mean'] > 0.95 else 'NO'} |
| Determinism (same seed) | identical | {'identical' if san['determinism_ok'] else 'DIFFERS'} | {'YES' if san['determinism_ok'] else 'NO'} |

The baseline and label-shuffle checks landing at ~0.5 confirm the pipeline is not
leaking; the leakage-ceiling check at ~1.0 confirms `account_status` was correctly
identified as a leak and excluded.

## Limitations

- **n = {paired['n_folds']} folds** is small. The paired t-test is a weak signal;
  the honest read is the overlap of the mean +/- sd intervals, not the p-value.
- The dataset's target is generated as a logistic function of the numeric features,
  which structurally favors a linear model; results may not generalize to datasets
  with strong feature interactions where boosting typically shines.
- `signup_date` carries no churn signal in this data, so the time-based split is a
  methodological safeguard rather than a source of distribution shift here.
- A single hold-out test set is not reported separately: the time-ordered CV folds
  serve as the evaluation, and no hyperparameters were tuned on them (default model
  settings), so no fold was used for selection.
"""
    return report


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run: python3 make_dataset.py --out churn.csv",
              file=sys.stderr)
        return 1

    raw = data_mod.load_raw(str(DATA_PATH))
    prepared = data_mod.prepare(raw)
    leaky_X = data_mod.build_leaky_features(raw).to_numpy()

    result = exp.compare(prepared, leaky_X)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(result, indent=2))

    # Flat per-fold CSV for quick inspection / downstream tooling.
    csv_lines = ["model,fold,roc_auc,average_precision,accuracy,f1,n_train,n_test,test_churn_rate"]
    for name, folds in result["per_fold"].items():
        for f in folds:
            csv_lines.append(
                f"{name},{f['fold']},{f['roc_auc']:.6f},{f['average_precision']:.6f},"
                f"{f['accuracy']:.6f},{f['f1']:.6f},{f['n_train']},{f['n_test']},"
                f"{f['test_churn_rate']:.6f}"
            )
    (RESULTS_DIR / "per_fold.csv").write_text("\n".join(csv_lines) + "\n")

    REPORT_PATH.write_text(write_report(result))

    print(exp.conclusion_text(result))
    print(f"\nWrote {RESULTS_DIR / 'metrics.json'}, {RESULTS_DIR / 'per_fold.csv'}, "
          f"and {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
