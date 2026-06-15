"""
Experiment: Does gradient boosting outperform logistic regression for churn?

Design:
- Variable: model family (LR vs GB); everything else held fixed.
- Split: TimeSeriesSplit(n_splits=5) over signup_date-sorted rows.
- Seeds: 5 per model → 25 evaluations each.
- Preprocessing: StandardScaler fitted on each train fold only.
- Primary metric: AUC-ROC (threshold-independent, handles imbalance).
- Secondary metric: F1.
- Winner criterion: gap > max(std_lr, std_gb); else "no detectable difference."
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.data import DATE_COL, FEATURE_COLS, TARGET_COL, load_and_clean
from src.pipeline import make_gb_pipeline, make_lr_pipeline

SEEDS = [42, 123, 456, 789, 1001]
N_SPLITS = 5
DATASET = "churn.csv"


def _fold_metrics(pipeline, X_tr, y_tr, X_te, y_te):
    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    y_pred = pipeline.predict(X_te)
    return {
        "auc_roc": float(roc_auc_score(y_te, y_prob)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
    }


def run_cv(df, make_fn, seeds=SEEDS, n_splits=N_SPLITS):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, f1s = [], []
    for seed in seeds:
        for tr_idx, te_idx in tscv.split(X):
            m = _fold_metrics(make_fn(random_state=seed),
                              X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            aucs.append(m["auc_roc"])
            f1s.append(m["f1"])
    return {
        "auc_roc_mean": float(np.mean(aucs)),
        "auc_roc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n": len(aucs),
    }


def run_baseline(df, n_splits=N_SPLITS):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, f1s = [], []
    for tr_idx, te_idx in tscv.split(X):
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X[tr_idx], y[tr_idx])
        y_pred = clf.predict(X[te_idx])
        y_prob = clf.predict_proba(X[te_idx])[:, 1]
        try:
            aucs.append(float(roc_auc_score(y[te_idx], y_prob)))
        except ValueError:
            aucs.append(0.5)
        f1s.append(float(f1_score(y[te_idx], y_pred, zero_division=0)))
    return {
        "auc_roc_mean": float(np.mean(aucs)),
        "auc_roc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n": len(aucs),
    }


def _conclude(lr, gb):
    gap = abs(lr["auc_roc_mean"] - gb["auc_roc_mean"])
    noise = max(lr["auc_roc_std"], gb["auc_roc_std"])
    if gap <= noise:
        return "no_detectable_difference", gap, noise
    winner = "gradient_boosting" if gb["auc_roc_mean"] > lr["auc_roc_mean"] else "logistic_regression"
    return f"{winner}_wins", gap, noise


def write_report(results):
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    base = results["baseline"]
    conclusion = results["conclusion"]
    gap = results["gap"]
    noise = results["noise_floor"]

    label = {
        "gradient_boosting_wins": "**Gradient boosting outperforms logistic regression.**",
        "logistic_regression_wins": "**Logistic regression outperforms gradient boosting.**",
        "no_detectable_difference": "**No detectable difference** between the two models (gap ≤ noise floor).",
    }[conclusion]

    report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Conclusion
{label}

| Model | AUC-ROC mean ± std | F1 mean ± std | n evals |
|---|---|---|---|
| Majority-class baseline | {base['auc_roc_mean']:.4f} ± {base['auc_roc_std']:.4f} | {base['f1_mean']:.4f} ± {base['f1_std']:.4f} | {base['n']} |
| Logistic Regression | {lr['auc_roc_mean']:.4f} ± {lr['auc_roc_std']:.4f} | {lr['f1_mean']:.4f} ± {lr['f1_std']:.4f} | {lr['n']} |
| Gradient Boosting | {gb['auc_roc_mean']:.4f} ± {gb['auc_roc_std']:.4f} | {gb['f1_mean']:.4f} ± {gb['f1_std']:.4f} | {gb['n']} |

AUC-ROC gap: {gap:.4f} | Noise floor (max std): {noise:.4f}

## Dataset
- Rows after deduplication: {results['dataset']['n_rows']}
- Churn rate: {results['dataset']['churn_rate']:.1%}
- Features used: {', '.join(results['dataset']['features_used'])}

## Methodology

### Leak Audit and Feature Selection
Three columns were excluded:

- **`customer_id`**: row identifier, zero predictive signal.
- **`signup_date`**: used to enforce temporal ordering of the split; encoding
  it as a numeric feature would confound cohort membership with model signal,
  so it is excluded from the feature matrix.
- **`days_since_last_login`**: **post-outcome leak.** A churned customer has,
  by definition, stopped logging in. This value is recorded *after* the outcome
  is known, not before. Including it would let the model read the answer from
  the data rather than learn a causal signal. It was dropped to ensure the
  pipeline generalises to the pre-churn decision window where this value is
  not yet observed.

### Deduplication
The dataset contains exact duplicate rows. These were removed before splitting
to prevent any duplicate from appearing in both train and test folds.

### Split Policy
`TimeSeriesSplit(n_splits={N_SPLITS})` over rows sorted by `signup_date`.
Each fold trains on earlier cohorts and tests on later ones — the operationally
realistic scenario where a model trained today predicts churn for future
customers.

### Preprocessing
`StandardScaler` is fitted on the train fold and applied to the test fold
within each cross-validation iteration. Test statistics never influence the
scaler.

### Repetition
{len(SEEDS)} random seeds × {N_SPLITS} CV folds = **{N_SPLITS * len(SEEDS)} evaluations per model**.
Mean ± std is reported. A winner is only claimed when the AUC-ROC gap exceeds
the noise floor (max std of the two models); otherwise the result is declared
"no detectable difference."

### Metrics
- **Primary: AUC-ROC** — threshold-independent, robust to class imbalance.
- **Secondary: F1** — at the default 0.5 threshold; included for completeness.

## Limitations
1. **Synthetic data**: results may not generalise to real churn datasets where
   behavioural sequences, product type, and lifecycle length add complexity.
2. **No hyperparameter tuning**: both models use fixed near-default settings.
   Tuned gradient boosting would likely show a larger advantage if one exists.
3. **F1 is threshold-dependent**: the default 0.5 threshold is arbitrary;
   AUC-ROC is the more reliable comparison metric here.
4. **Short observation window**: all customers in this dataset signed up within
   ~2.5 years; longer temporal drift may alter the comparison.
"""
    Path("REPORT.md").write_text(report)


def main():
    if not Path(DATASET).exists():
        print(f"Dataset not found. Run: python3 make_dataset.py --out {DATASET}")
        sys.exit(1)

    print(f"Loading {DATASET}...")
    df = load_and_clean(DATASET)
    churn_rate = df[TARGET_COL].mean()
    print(f"  {len(df)} rows, churn rate: {churn_rate:.1%}, features: {FEATURE_COLS}")

    print(f"\nRunning {N_SPLITS}-fold TimeSeriesSplit × {len(SEEDS)} seeds...")
    print("  baseline...")
    base = run_baseline(df)
    print("  logistic regression...")
    lr = run_cv(df, make_lr_pipeline)
    print("  gradient boosting...")
    gb = run_cv(df, make_gb_pipeline)

    conclusion, gap, noise = _conclude(lr, gb)

    results = {
        "dataset": {
            "n_rows": len(df),
            "churn_rate": float(churn_rate),
            "features_used": FEATURE_COLS,
            "features_dropped": {
                "customer_id": "identifier",
                "signup_date": "used for split ordering only",
                "days_since_last_login": "post-outcome leak",
            },
        },
        "methodology": {
            "split": f"TimeSeriesSplit(n_splits={N_SPLITS}), sorted by signup_date",
            "seeds": SEEDS,
            "n_evals_per_model": N_SPLITS * len(SEEDS),
            "preprocessing": "StandardScaler fitted on train fold only",
            "primary_metric": "AUC-ROC",
        },
        "baseline": base,
        "logistic_regression": lr,
        "gradient_boosting": gb,
        "conclusion": conclusion,
        "gap": float(gap),
        "noise_floor": float(noise),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    write_report(results)

    print(f"\n{'='*50}")
    print(f"  Baseline AUC-ROC:   {base['auc_roc_mean']:.4f} ± {base['auc_roc_std']:.4f}")
    print(f"  LR  AUC-ROC:        {lr['auc_roc_mean']:.4f} ± {lr['auc_roc_std']:.4f}")
    print(f"  GB  AUC-ROC:        {gb['auc_roc_mean']:.4f} ± {gb['auc_roc_std']:.4f}")
    print(f"  Gap: {gap:.4f}  |  Noise floor: {noise:.4f}")
    print(f"  Conclusion: {conclusion}")
    print(f"{'='*50}")
    print("\nWrote: results/metrics.json, REPORT.md")


if __name__ == "__main__":
    main()
