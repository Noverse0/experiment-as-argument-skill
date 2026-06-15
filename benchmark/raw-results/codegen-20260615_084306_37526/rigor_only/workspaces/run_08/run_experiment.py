"""Entrypoint: run the full churn experiment and write results + report."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import load_and_clean, make_models, FEATURE_COLS
from src.evaluate import evaluate_models, paired_ttest

SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5
DATA_PATH = "churn.csv"


def sanity_checks(X, y, dropped_dupes: int) -> None:
    assert len(X) > 0, "Dataset is empty after cleaning"
    assert y.nunique() == 2, "Target must be binary"
    assert dropped_dupes >= 0, "Negative duplicate count"
    churn_rate = y.mean()
    assert 0.01 < churn_rate < 0.99, f"Degenerate class balance: {churn_rate:.3f}"
    print(f"  Rows after dedup: {len(X)}  (dropped {dropped_dupes} exact duplicates)")
    print(f"  Churn rate: {churn_rate:.3f}")
    print(f"  Features: {list(X.columns)}")


def main() -> None:
    # Step 1: generate dataset if absent
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        subprocess.run(
            [sys.executable, "make_dataset.py", "--out", DATA_PATH], check=True
        )

    print("\n=== Churn Experiment: GB vs LR ===\n")

    # Step 2: load and clean
    X, y, dropped_dupes = load_and_clean(DATA_PATH)
    print("Data checks:")
    sanity_checks(X, y, dropped_dupes)

    # Step 3: evaluate
    print(f"\nRunning {N_SPLITS}-fold CV × {len(SEEDS)} seeds = "
          f"{N_SPLITS * len(SEEDS)} evaluations per model...")
    models = make_models()
    results = evaluate_models(X, y, models, n_splits=N_SPLITS, seeds=SEEDS)

    lr = results["LogisticRegression"]
    gb = results["GradientBoosting"]
    t_stat, p_value = paired_ttest(lr["roc_auc_all"], gb["roc_auc_all"])

    # Step 4: persist machine-readable metrics
    os.makedirs("results", exist_ok=True)
    output = {
        "config": {
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "n_evaluations_per_model": lr["n_evaluations"],
            "features": FEATURE_COLS,
            "excluded_features": {
                "customer_id": "identifier",
                "signup_date": "temporal, no deployment anchor",
                "days_since_last_login": "target leak: derived from churn outcome",
            },
        },
        "models": results,
        "comparison": {
            "gb_minus_lr_roc_auc": round(gb["roc_auc_mean"] - lr["roc_auc_mean"], 6),
            "paired_ttest_t": round(t_stat, 4),
            "paired_ttest_p": round(p_value, 4),
            "significant_at_0.05": p_value < 0.05,
        },
    }
    with open("results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    # Step 5: write report
    gap = gb["roc_auc_mean"] - lr["roc_auc_mean"]
    winner = "GradientBoosting" if gap > 0 else "LogisticRegression"
    significant = p_value < 0.05

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

### Variable
Model class — everything else held fixed: same features, same preprocessing (StandardScaler),
same CV folds, same random seeds.

### Features Used
| Feature | Justification |
|---------|--------------|
| `tenure_months` | Customer age; genuine causal signal |
| `monthly_spend` | Revenue proxy; included in DGP signal |
| `support_tickets` | Dissatisfaction proxy; included in DGP signal |

### Excluded Features (Leak Audit)
| Feature | Reason |
|---------|--------|
| `customer_id` | Row identifier — no signal |
| `signup_date` | Temporal column; without a deployment time anchor a random split would be invalid for time-based features |
| `days_since_last_login` | **Target leak** — churned customers stop logging in, so this value is recorded *after* the outcome is known. Including it would inflate AUC artificially without being available at prediction time. |

### Data Cleaning
Removed **{dropped_dupes} exact duplicate rows** before any split. Duplicates straddling
train/test in a random split would inflate held-out metrics.

### Evaluation Protocol
- {N_SPLITS}-fold stratified cross-validation repeated over {len(SEEDS)} seeds: `{SEEDS}`
- Total fold evaluations per model: **{lr["n_evaluations"]}**
- Primary metric: **ROC-AUC** (threshold-free, handles class imbalance)
- Secondary metric: **F1** (threshold-sensitive summary)
- Significance: paired t-test on fold-level AUC scores (df = {lr["n_evaluations"] - 1})

### Class Balance
Churn rate in cleaned dataset: **{y.mean():.1%}**

## Results

| Model | ROC-AUC mean ± std | F1 mean ± std |
|-------|--------------------|---------------|
| LogisticRegression | {lr["roc_auc_mean"]:.4f} ± {lr["roc_auc_std"]:.4f} | {lr["f1_mean"]:.4f} ± {lr["f1_std"]:.4f} |
| GradientBoosting | {gb["roc_auc_mean"]:.4f} ± {gb["roc_auc_std"]:.4f} | {gb["f1_mean"]:.4f} ± {gb["f1_std"]:.4f} |

**AUC gap (GB − LR): {gap:+.4f}**
Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f} ({"significant" if significant else "not significant"} at α = 0.05)

## Conclusion

{"**No detectable difference.** " if not significant else f"**{winner} wins.** "}{"The gap between models (" + f"{abs(gap):.4f} AUC) is within noise; the paired t-test does not reject H₀ (p = " + f"{p_value:.4f}). The honest conclusion is that neither model is reliably better than the other on this dataset and feature set." if not significant else f"GradientBoosting achieves higher AUC by {abs(gap):.4f} (p = {p_value:.4f} < 0.05). The difference is statistically detectable across {lr['n_evaluations']} fold evaluations."}

## Limitations

1. **Temporal validity not tested**: `signup_date` was excluded because a random CV split
   cannot simulate a time-ordered deployment. A production evaluation should train on
   customers who signed up before a cutoff and test on those who signed up after.

2. **No hyperparameter tuning**: GradientBoosting uses default hyperparameters. A tuned
   GB may perform differently relative to tuned LR. Tuning budget should be identical
   for both arms to keep the comparison fair.

3. **Synthetic data**: The DGP uses a simple logistic signal (tenure, spend, tickets)
   that LR is well-specified for. Results may not generalize to real churn datasets
   with non-linear interactions.

4. **Single dataset version**: Results hold for seed=7, n=4200. A different dataset
   realization could shift the ranking.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)

    # Step 6: print summary
    print(f"\nResults:")
    print(f"  LR  ROC-AUC: {lr['roc_auc_mean']:.4f} ± {lr['roc_auc_std']:.4f}")
    print(f"  GB  ROC-AUC: {gb['roc_auc_mean']:.4f} ± {gb['roc_auc_std']:.4f}")
    print(f"  Gap (GB-LR): {gap:+.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f} "
          f"({'significant' if significant else 'not significant'} at α=0.05)")
    print(f"\nConclusion: {'No detectable difference' if not significant else f'{winner} outperforms'}")
    print(f"\nArtifacts written: results/metrics.json, REPORT.md")


if __name__ == "__main__":
    main()
