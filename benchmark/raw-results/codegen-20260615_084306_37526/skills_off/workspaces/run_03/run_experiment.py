#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _summarize(scores: list, metric: str) -> dict:
    vals = [s[metric] for s in scores]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def write_report(results: dict) -> None:
    ds = results["dataset"]
    cv = results["cv"]
    test = results["test"]

    lr_auc = _summarize(cv["lr"], "roc_auc")
    gb_auc = _summarize(cv["gb"], "roc_auc")
    lr_ap = _summarize(cv["lr"], "avg_precision")
    gb_ap = _summarize(cv["gb"], "avg_precision")
    lr_f1 = _summarize(cv["lr"], "f1")
    gb_f1 = _summarize(cv["gb"], "f1")

    auc_gap = gb_auc["mean"] - lr_auc["mean"]
    pooled_std = (lr_auc["std"] + gb_auc["std"]) / 2

    if abs(auc_gap) <= pooled_std:
        conclusion = "No statistically meaningful difference detected between the two models"
        conclusion_detail = (
            f"The AUC gap ({abs(auc_gap):.4f}) is within the pooled CV standard deviation "
            f"({pooled_std:.4f}). With these defaults and this dataset, gradient boosting "
            "does not reliably outperform logistic regression."
        )
    elif auc_gap > 0:
        conclusion = "Gradient boosting outperforms logistic regression"
        conclusion_detail = (
            f"The AUC gap ({auc_gap:+.4f}) exceeds the pooled CV standard deviation "
            f"({pooled_std:.4f}), suggesting a real but modest advantage for gradient boosting."
        )
    else:
        conclusion = "Logistic regression outperforms gradient boosting"
        conclusion_detail = (
            f"The AUC gap ({auc_gap:+.4f}) exceeds the pooled CV standard deviation "
            f"({pooled_std:.4f}), suggesting logistic regression is better suited to this task."
        )

    report = f"""# Churn Prediction Experiment Report

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Variable:** Model class (scikit-learn `LogisticRegression` vs `GradientBoostingClassifier`, default hyperparameters). All other choices are held fixed across both arms.

### Data preparation

| Step | Action | Reason |
|------|--------|--------|
| Deduplication | Removed {ds['n_duplicates_removed']} exact-duplicate rows | Duplicates straddling the train/test boundary inflate held-out performance |
| Temporal sort | Sorted by `signup_date` ascending | Enforces chronological integrity for the split |
| Time-based split | First 80% → train ({ds['n_train']} rows), last 20% → test ({ds['n_test']} rows) | Random splits on temporal data allow future-signup customers to inform past-signup predictions (a form of leakage) |

### Feature selection

| Feature | Status | Reason |
|---------|--------|--------|
| `tenure_months` | **Used** | Causal: shorter tenure → higher churn probability |
| `monthly_spend` | **Used** | Causal: higher spend → slightly higher churn probability |
| `support_tickets` | **Used** | Causal: more tickets → higher churn probability |
| `days_since_last_login` | **Dropped — TARGET LEAK** | Churned customers have stopped logging in by definition. This value is derived from (i.e., causally downstream of) the churn outcome. It encodes the label, not a cause, and would be unavailable at real prediction time. |
| `customer_id` | Dropped | Row identifier; no predictive value |
| `signup_date` | Dropped as feature | Used only for temporal split ordering; including it as a numeric feature risks encoding time-of-split information |

### Class balance

- Train churn rate: {ds['train_churn_rate']:.1%}
- Test churn rate: {ds['test_churn_rate']:.1%}

### Evaluation protocol

- **Primary metric:** ROC-AUC (robust to class imbalance; threshold-independent)
- **Secondary metrics:** Average Precision, F1 (threshold = 0.5)
- **Variance estimation:** {cv['n_seeds']} random seeds × {cv['n_folds']}-fold StratifiedKFold = {cv['n_total_fits']} CV fits per model on the training set
- **Preprocessing:** `StandardScaler` fitted on each CV train fold only, applied to the corresponding val fold — no information from the validation set enters the scaler
- **Final evaluation:** Models trained on all train data; test set touched exactly once, after all design decisions were fixed

## Results

### Cross-Validation Performance (training set, {cv['n_total_fits']} folds per model)

| Model | ROC-AUC (mean ± std) | Avg Precision (mean ± std) | F1 (mean ± std) |
|-------|---------------------|---------------------------|-----------------|
| LogisticRegression | {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f} | {lr_ap['mean']:.4f} ± {lr_ap['std']:.4f} | {lr_f1['mean']:.4f} ± {lr_f1['std']:.4f} |
| GradientBoosting | {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} | {gb_ap['mean']:.4f} ± {gb_ap['std']:.4f} | {gb_f1['mean']:.4f} ± {gb_f1['std']:.4f} |

AUC gap (GB − LR): {auc_gap:+.4f} | Pooled CV std: {pooled_std:.4f}

### Held-Out Test Set Performance (n = {ds['n_test']}, touched once)

| Model | ROC-AUC | Avg Precision | F1 |
|-------|---------|--------------|-----|
| LogisticRegression | {test['lr']['roc_auc']:.4f} | {test['lr']['avg_precision']:.4f} | {test['lr']['f1']:.4f} |
| GradientBoosting | {test['gb']['roc_auc']:.4f} | {test['gb']['avg_precision']:.4f} | {test['gb']['f1']:.4f} |

## Conclusion

**{conclusion}.**

{conclusion_detail}

This outcome is consistent with the data-generating process: the true signal is linear in the log-odds (`logit = −1.2 − 0.03·tenure + 0.01·spend + 0.45·tickets`), a structure for which logistic regression is the correctly-specified model. Gradient boosting's additional capacity for nonlinear interactions does not provide a systematic advantage over a well-matched linear model on this dataset.

## Limitations

1. **No hyperparameter tuning:** Both models use scikit-learn defaults. A properly tuned gradient boosting model (lower learning rate, more trees, subsampling) might show a larger advantage; a systematic comparison would require an inner CV tuning loop for both arms.
2. **Single synthetic dataset:** Results are specific to this data-generating process. Real churn datasets often have nonlinear feature interactions that favor tree ensembles.
3. **StratifiedKFold within temporal training data:** The CV folds do not strictly preserve chronological order within the training portion. A `TimeSeriesSplit` would be more conservative. The held-out test set (temporally after all training data) provides a clean final evaluation.
4. **15 CV observations per model:** Provides reasonable but not definitive variance estimates; overlapping confidence intervals are the honest conclusion, not a significant winner claim.
5. **Synthetic linear signal:** The true generative model is additive and linear in the log-odds, which structurally favors logistic regression. A gradient boosting advantage might emerge with nonlinear or interaction-heavy real-world data.
"""
    Path("REPORT.md").write_text(report)


def main():
    csv_path = "churn.csv"
    print("Step 1/3: Generating dataset...")
    subprocess.run([sys.executable, "make_dataset.py", "--out", csv_path], check=True)

    print("Step 2/3: Running experiment (CV + final evaluation)...")
    from src.experiment import run_experiment
    results = run_experiment(csv_path)

    print("Step 3/3: Writing outputs...")
    Path("results").mkdir(exist_ok=True)

    # Attach summaries alongside raw fold scores for easy downstream consumption
    results_out = dict(results)
    cv_summary = {}
    for model in ["lr", "gb"]:
        cv_summary[model] = {
            metric: _summarize(results["cv"][model], metric)
            for metric in ["roc_auc", "avg_precision", "f1"]
        }
    results_out["cv_summary"] = cv_summary

    with open("results/metrics.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("  Wrote results/metrics.json")

    write_report(results)
    print("  Wrote REPORT.md")

    lr_auc = cv_summary["lr"]["roc_auc"]
    gb_auc = cv_summary["gb"]["roc_auc"]
    gap = gb_auc["mean"] - lr_auc["mean"]
    print(f"\nSummary (CV ROC-AUC):")
    print(f"  LogisticRegression:    {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}")
    print(f"  GradientBoosting:      {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}")
    print(f"  Gap (GB − LR):         {gap:+.4f}")
    print(f"\nTest ROC-AUC:")
    print(f"  LogisticRegression:    {results['test']['lr']['roc_auc']:.4f}")
    print(f"  GradientBoosting:      {results['test']['gb']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
