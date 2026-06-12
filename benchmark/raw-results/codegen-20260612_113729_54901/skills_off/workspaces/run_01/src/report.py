"""Generate REPORT.md from experiment summary."""
from pathlib import Path


def _overlap(a_mean, a_std, b_mean, b_std) -> bool:
    """Return True if the ±1 SD intervals overlap."""
    return (a_mean - a_std) <= (b_mean + b_std) and (b_mean - b_std) <= (a_mean + a_std)


def _conclusion(lr: dict, gb: dict) -> str:
    lr_mean = lr["roc_auc"]["mean"]
    lr_std = lr["roc_auc"]["std"]
    gb_mean = gb["roc_auc"]["mean"]
    gb_std = gb["roc_auc"]["std"]
    gap = gb_mean - lr_mean

    if _overlap(lr_mean, lr_std, gb_mean, gb_std):
        direction = "higher" if gap > 0 else "lower"
        return (
            f"Gradient boosting scores {abs(gap):.3f} ROC-AUC {direction} than logistic "
            f"regression, but the ±1 SD intervals overlap "
            f"(LR {lr_mean:.3f}±{lr_std:.3f}, GB {gb_mean:.3f}±{gb_std:.3f}). "
            f"**No detectable difference** between the two models on this dataset."
        )
    elif gap > 0:
        return (
            f"Gradient boosting outperforms logistic regression by {gap:.3f} ROC-AUC "
            f"(LR {lr_mean:.3f}±{lr_std:.3f}, GB {gb_mean:.3f}±{gb_std:.3f}, "
            f"non-overlapping ±1 SD intervals)."
        )
    else:
        return (
            f"Logistic regression matches or exceeds gradient boosting "
            f"(LR {lr_mean:.3f}±{lr_std:.3f}, GB {gb_mean:.3f}±{gb_std:.3f}, "
            f"non-overlapping ±1 SD intervals in favour of LR)."
        )


def write_report(summary: dict, n_rows: int, churn_rate: float, n_dups_removed: int,
                 sanity: dict) -> None:
    lr = summary["logistic_regression"]
    gb = summary["gradient_boosting"]
    conclusion = _conclusion(lr, gb)
    seeds = summary["seeds"]
    n_splits = summary["n_splits"]

    shuffle_auc_lr = sanity.get("lr_shuffle_auc", "n/a")
    shuffle_auc_gb = sanity.get("gb_shuffle_auc", "n/a")
    baseline_auc = sanity.get("baseline_auc", "n/a")

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn
on this dataset?

## Methodology

**Variable:** Model class (LogisticRegression vs GradientBoostingClassifier).
All preprocessing, split policy, seeds, and hyperparameter budgets are held constant.

### Data Cleaning (Rigor Decisions)

| Step | Rationale |
|------|-----------|
| Drop `account_status` | Perfectly correlated with `churned` (closed ↔ churned=1) — direct label leakage. |
| Drop `customer_id` | Row identifier; carries no predictive signal. |
| Deduplicate before split | {n_dups_removed} exact duplicate rows removed; without this, duplicates straddle train/test and inflate metrics. |
| Convert `signup_date` → `signup_days` | Encodes temporal position as an integer for use in a time-aware split. |
| Sort by `signup_days` | Required so `TimeSeriesSplit` allocates earlier data to training and later data to test — preventing future leakage from a random split. |

### Split Policy

`TimeSeriesSplit(n_splits={n_splits})` on the sorted dataset. Each fold trains on the
earliest slice and tests on the next chronological slice — no future data ever appears
in training.

### Preprocessing

`StandardScaler` is fitted on the training fold only and applied to the test fold.
No fit-transform is applied to the full dataset before splitting.

### Repetition

Experiment repeated over seeds `{seeds}` to measure variance from model randomness.
Each seed × fold pair is an independent evaluation point
({len(seeds)} seeds × {n_splits} folds = {len(seeds) * n_splits} total evaluations per model).
Results below are aggregated means ± SD across seeds.

### Metrics

- **Primary:** ROC-AUC — threshold-free, robust to class imbalance (churn rate: {churn_rate:.1%}).
- **Secondary:** F1 (at default 0.5 threshold), PR-AUC.

---

## Sanity Checks

| Check | LR | GB | Pass? |
|-------|----|----|-------|
| Majority-class baseline AUC | {baseline_auc:.3f} | — | models must exceed this |
| Label-shuffle AUC (≈0.5 expected) | {shuffle_auc_lr:.3f} | {shuffle_auc_gb:.3f} | {'PASS' if isinstance(shuffle_auc_lr, float) and shuffle_auc_lr < 0.6 else 'WARN'} / {'PASS' if isinstance(shuffle_auc_gb, float) and shuffle_auc_gb < 0.6 else 'WARN'} |

Label-shuffle AUC near 0.5 confirms no label-independent signal is leaking through
the feature set (e.g., a missed leaky column).

---

## Results

**Dataset after cleaning:** {n_rows} rows, 4 features
(`tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`)

| Model | ROC-AUC mean ± SD | F1 mean ± SD | PR-AUC mean ± SD |
|---|---|---|---|
| Logistic Regression | {lr["roc_auc"]["mean"]:.3f} ± {lr["roc_auc"]["std"]:.3f} | {lr["f1"]["mean"]:.3f} ± {lr["f1"]["std"]:.3f} | {lr["pr_auc"]["mean"]:.3f} ± {lr["pr_auc"]["std"]:.3f} |
| Gradient Boosting | {gb["roc_auc"]["mean"]:.3f} ± {gb["roc_auc"]["std"]:.3f} | {gb["f1"]["mean"]:.3f} ± {gb["f1"]["std"]:.3f} | {gb["pr_auc"]["mean"]:.3f} ± {gb["pr_auc"]["std"]:.3f} |

*Aggregated over {len(seeds)} seeds × {n_splits} folds.*

---

## Conclusion

{conclusion}

---

## Limitations

- **Synthetic data:** The dataset is generated from a logistic model with additive noise.
  Real churn data typically has non-linear interactions not present here — results may
  not generalise.
- **No hyperparameter tuning:** Both models use near-default parameters with a fixed
  `n_estimators=100` for GB. A tuned GB on a real dataset could show a larger or
  smaller gap.
- **Small feature set:** Only 4 features survive after dropping leakage. On a richer
  feature set, the relative advantage of non-linear models typically grows.
- **Seeds × folds count:** {len(seeds) * n_splits} evaluation points is sufficient for
  rough comparison but not for formal statistical significance testing.
- **Time axis proxy:** `signup_date` is used as the temporal axis. In a real deployment,
  the relevant axis is the prediction date relative to the observation window.
"""

    Path("REPORT.md").write_text(report)
    print("Wrote REPORT.md")
