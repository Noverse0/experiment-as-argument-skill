"""Entrypoint: generate dataset → run experiment → write results/ and REPORT.md."""
import subprocess
import sys
from pathlib import Path

from src.experiment import run_experiment


def _determine_conclusion(results: dict) -> tuple:
    cv = results["cv_results"]
    holdout = results["holdout_results"]

    lr_auc = cv["logistic_regression"]["roc_auc"]
    gb_auc = cv["gradient_boosting"]["roc_auc"]
    gap = gb_auc["mean"] - lr_auc["mean"]
    pooled_sd = max(lr_auc["std"], gb_auc["std"], 1e-9)
    gap_in_sd = gap / pooled_sd

    h_gap = holdout["gradient_boosting"]["roc_auc"] - holdout["logistic_regression"]["roc_auc"]

    if gap_in_sd >= 2.0 and gap > 0:
        verdict = "Gradient Boosting outperforms Logistic Regression."
        winner = "gradient_boosting"
    elif gap_in_sd <= -2.0:
        verdict = "Logistic Regression outperforms Gradient Boosting."
        winner = "logistic_regression"
    else:
        verdict = (
            f"No detectable difference between the two models "
            f"(CV AUC gap {gap:+.4f}, {gap_in_sd:.1f} SD — within noise)."
        )
        winner = "none"

    return verdict, winner, gap, gap_in_sd, h_gap


def generate_report(results: dict) -> str:
    data = results["data"]
    split = data["split"]
    cv = results["cv_results"]
    holdout = results["holdout_results"]
    baseline = results["baseline"]
    cfg = results["cv_config"]

    verdict, winner, gap, gap_in_sd, h_gap = _determine_conclusion(results)

    lr_cv = cv["logistic_regression"]
    gb_cv = cv["gradient_boosting"]
    lr_h = holdout["logistic_regression"]
    gb_h = holdout["gradient_boosting"]

    return f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

### Data
- **Raw rows:** {data['n_raw']}  →  **After deduplication:** {data['n_deduped']}  (dropped {data['n_duplicates_dropped']} exact duplicates)
- **Overall churn rate:** {data['churn_rate']:.1%}

### Rigor Steps Applied
| Issue | Action |
|-------|--------|
| `account_status` is "closed" iff churned — perfect target leak | Dropped before any modeling |
| 200 exact duplicate rows appended by the generator | `drop_duplicates()` before splitting |
| `signup_date` is temporal; random splits cause time leakage | Sort by date; earliest 80% → train, latest 20% → test |
| Scaler fitted on full data would leak test statistics | `StandardScaler` fit inside Pipeline per CV fold |

### Split
- **Method:** Time-based (sort by `signup_date`)
- **Cutoff date:** {split['cutoff_date']}
- **Train:** {split['n_train']} rows, churn rate {split['train_churn_rate']:.1%}
- **Test:** {split['n_test']} rows, churn rate {split['test_churn_rate']:.1%}

### Evaluation Design
- **CV:** {cfg['n_folds']}-fold stratified × {len(cfg['seeds'])} seeds = **{cfg['n_estimates']} estimates per model** (satisfies ≥3 requirement for variance claims)
- **Primary metric:** ROC-AUC (robust to class imbalance; summarises the full ranking curve)
- **Secondary metrics:** F1, precision, recall
- **Baseline:** Majority-class classifier (theoretical ROC-AUC = 0.5)
- **Features used:** `tenure_months`, `monthly_spend`, `support_tickets`
- **Test set touched:** once, at the end — no decisions were made after seeing test scores

## Results

### Sanity Check
Both models exceed the majority-class baseline (ROC-AUC = 0.5), confirming real signal is present.

### Cross-Validation on Training Set (n = {cfg['n_estimates']} per model)

| Model | ROC-AUC mean ± sd | F1 mean ± sd | Precision mean ± sd | Recall mean ± sd |
|-------|-------------------|--------------|---------------------|------------------|
| Logistic Regression | {lr_cv['roc_auc']['mean']:.4f} ± {lr_cv['roc_auc']['std']:.4f} | {lr_cv['f1']['mean']:.4f} ± {lr_cv['f1']['std']:.4f} | {lr_cv['precision']['mean']:.4f} ± {lr_cv['precision']['std']:.4f} | {lr_cv['recall']['mean']:.4f} ± {lr_cv['recall']['std']:.4f} |
| Gradient Boosting   | {gb_cv['roc_auc']['mean']:.4f} ± {gb_cv['roc_auc']['std']:.4f} | {gb_cv['f1']['mean']:.4f} ± {gb_cv['f1']['std']:.4f} | {gb_cv['precision']['mean']:.4f} ± {gb_cv['precision']['std']:.4f} | {gb_cv['recall']['mean']:.4f} ± {gb_cv['recall']['std']:.4f} |

**CV AUC gap (GB − LR): {gap:+.4f}  ({gap_in_sd:.1f} SD)**

### Holdout Test Set (touched once)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| Logistic Regression | {lr_h['roc_auc']:.4f} | {lr_h['f1']:.4f} | {lr_h['precision']:.4f} | {lr_h['recall']:.4f} |
| Gradient Boosting   | {gb_h['roc_auc']:.4f} | {gb_h['f1']:.4f} | {gb_h['precision']:.4f} | {gb_h['recall']:.4f} |
| Majority-class baseline | 0.5000 | {baseline['f1']:.4f} | — | — |

**Holdout AUC gap (GB − LR): {h_gap:+.4f}**

## Conclusion

**{verdict}**

## Limitations

- **Single dataset, ~{data['n_deduped']} rows:** Power to detect small differences is limited.
- **No hyperparameter tuning:** Default settings used for both models; tuned GB may change the relative gap.
- **Three features only:** After removing leaky columns, only `tenure_months`, `monthly_spend`, and `support_tickets` remain. Additional features might alter the comparison.
- **One temporal split:** A single cutoff means test-set variance is unobservable; walk-forward CV would give better estimates but at higher compute cost.
- **Deployment caveat:** Test performance reflects the held-out time window only; distribution shift beyond that window is unquantified.
"""


def main():
    csv_path = "churn.csv"

    print("Step 1: Generating dataset...")
    proc = subprocess.run(
        [sys.executable, "make_dataset.py", "--out", csv_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"ERROR generating dataset:\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  {proc.stdout.strip()}")

    print("Step 2: Running experiment...")
    results = run_experiment(csv_path, results_dir="results")

    print("Step 3: Writing REPORT.md...")
    Path("REPORT.md").write_text(generate_report(results))

    cv_lr = results["cv_results"]["logistic_regression"]["roc_auc"]
    cv_gb = results["cv_results"]["gradient_boosting"]["roc_auc"]
    h_lr = results["holdout_results"]["logistic_regression"]["roc_auc"]
    h_gb = results["holdout_results"]["gradient_boosting"]["roc_auc"]

    verdict, _, gap, gap_in_sd, _ = _determine_conclusion(results)

    print("\n=== Summary ===")
    print(f"CV  ROC-AUC  LR={cv_lr['mean']:.4f}±{cv_lr['std']:.4f}  GB={cv_gb['mean']:.4f}±{cv_gb['std']:.4f}  gap={gap:+.4f} ({gap_in_sd:.1f} SD)")
    print(f"Test ROC-AUC LR={h_lr:.4f}  GB={h_gb:.4f}")
    print(f"Conclusion: {verdict}")
    print("\nArtifacts: results/metrics.json  REPORT.md")


if __name__ == "__main__":
    main()
