"""Entrypoint: run the full churn experiment and write results + REPORT.md."""
import json
import sys
from pathlib import Path

from src.experiment import run_experiment


def write_report(results: dict, path: str = "REPORT.md") -> None:
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    c = results["conclusion"]
    sanity = results["sanity"]

    verdict_text = {
        "gradient_boosting_wins": (
            "Gradient Boosting outperforms Logistic Regression with a gap "
            f"({c['gap']:.4f} AUC) exceeding the noise threshold ({c['noise_threshold']:.4f})."
        ),
        "logistic_regression_wins": (
            "Logistic Regression outperforms Gradient Boosting with a gap "
            f"({c['gap']:.4f} AUC) exceeding the noise threshold ({c['noise_threshold']:.4f})."
        ),
        "no_detectable_difference": (
            "No detectable difference: the AUC gap between models "
            f"({c['gap']:.4f}) does not exceed the noise threshold ({c['noise_threshold']:.4f}). "
            "The honest conclusion is that neither model is reliably better on this dataset."
        ),
    }[c["verdict"]]

    warn_block = ""
    if sanity.get("warnings"):
        warn_block = "\n**Sanity warnings:**\n" + "\n".join(
            f"- {w}" for w in sanity["warnings"]
        ) + "\n"

    report = f"""# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Conclusion

**{c['verdict'].replace('_', ' ').title()}**

{verdict_text}

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| LogisticRegression | {lr['roc_auc']['mean']:.4f} ± {lr['roc_auc']['std']:.4f} | {lr['f1']['mean']:.4f} ± {lr['f1']['std']:.4f} |
| GradientBoostingClassifier | {gb['roc_auc']['mean']:.4f} ± {gb['roc_auc']['std']:.4f} | {gb['f1']['mean']:.4f} ± {gb['f1']['std']:.4f} |

Seeds: {results['seeds']} ({results['n_seeds']} runs per model)

## Methodology

**Claim being tested:** Does GradientBoostingClassifier outperform LogisticRegression
for predicting customer churn on this dataset?

**Variable:** Model class. All other choices (features, split, hyperparameters) are fixed.

**Data preparation:**
- Removed {4200 - results['train_size'] - results['test_size']} exact duplicate rows *before* splitting
  to prevent train/test contamination via memorised duplicates.
- Dropped `account_status`: it is derived directly from the target (`"closed"` iff `churned==1`),
  making it a perfect-leakage feature.
- Dropped `customer_id`: row identifier, not predictive.
- Converted `signup_date` to ordinal days as a numeric feature.

**Split:** Temporal (time-ordered by `signup_date`, 80/20).
Customers who signed up later form the test set, matching realistic deployment where
a model trained on historical data is applied to new customers.

- Train: {results['train_size']} rows (churn rate {results['train_churn_rate']:.1%})
- Test: {results['test_size']} rows (churn rate {results['test_churn_rate']:.1%})

**Metrics:** Primary = ROC-AUC (threshold-independent, robust to class imbalance at 27% positive rate).
Secondary = F1, precision, recall, accuracy.

**Variance:** 3 random seeds vary model-internal randomness; data split is fixed.
Winner requires AUC gap > max(per-model std) to avoid noise-driven claims.

## Sanity Check Results

- Baseline (majority class) AUC: {sanity['baseline_auc']:.4f} (expected ~0.5)
- Overfit tiny subset: {"PASS" if sanity['overfit_ok'] else "FAIL"}
- Label-shuffle AUC: {sanity['shuffle_auc']:.4f} (expected ~0.5)
{warn_block}
## Detailed Results

### Logistic Regression

| Metric | Mean | Std | Runs |
|---|---|---|---|
| ROC-AUC | {lr['roc_auc']['mean']:.4f} | {lr['roc_auc']['std']:.4f} | {lr['roc_auc']['runs']} |
| F1 | {lr['f1']['mean']:.4f} | {lr['f1']['std']:.4f} | {lr['f1']['runs']} |
| Precision | {lr['precision']['mean']:.4f} | {lr['precision']['std']:.4f} | {lr['precision']['runs']} |
| Recall | {lr['recall']['mean']:.4f} | {lr['recall']['std']:.4f} | {lr['recall']['runs']} |
| Accuracy | {lr['accuracy']['mean']:.4f} | {lr['accuracy']['std']:.4f} | {lr['accuracy']['runs']} |

### Gradient Boosting Classifier

| Metric | Mean | Std | Runs |
|---|---|---|---|
| ROC-AUC | {gb['roc_auc']['mean']:.4f} | {gb['roc_auc']['std']:.4f} | {gb['roc_auc']['runs']} |
| F1 | {gb['f1']['mean']:.4f} | {gb['f1']['std']:.4f} | {gb['f1']['runs']} |
| Precision | {gb['precision']['mean']:.4f} | {gb['precision']['std']:.4f} | {gb['precision']['runs']} |
| Recall | {gb['recall']['mean']:.4f} | {gb['recall']['std']:.4f} | {gb['recall']['runs']} |
| Accuracy | {gb['accuracy']['mean']:.4f} | {gb['accuracy']['std']:.4f} | {gb['accuracy']['runs']} |

## Limitations

- **3 seeds is a lower bound on variance** — more seeds or cross-validation would tighten estimates.
- **Fixed hyperparameters** — GB with tuned depth/estimators might differ; LR with C tuning likewise.
  Comparing untuned models is conservative but fair given equal tuning budget (none).
- **Single train/test split** — a rolling temporal CV would give a more stable estimate.
- **Features are few and numeric** — the dataset is synthetic and small; results may not
  generalise to richer real-world churn data.
"""
    Path(path).write_text(report)
    print(f"\nReport written to {path}")


def main():
    data_path = "churn.csv"
    if not Path(data_path).exists():
        print(f"Dataset not found. Run: python3 make_dataset.py --out {data_path}")
        sys.exit(1)

    results = run_experiment(data_path)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Metrics written to {metrics_path}")

    write_report(results)


if __name__ == "__main__":
    main()
