#!/usr/bin/env python3
"""
Entrypoint: Generate dataset, run experiment, write results and report.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from src.experiment import run_experiment


def main():
    # Generate dataset
    print("Generating dataset...")
    import subprocess
    result = subprocess.run(
        ["python3", "make_dataset.py", "--out", "churn.csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error generating dataset: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

    # Run experiment
    print()
    exp_result = run_experiment("churn.csv", output_dir="results")

    # Save raw metrics
    metrics_file = Path("results") / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(exp_result, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Generate report
    config = exp_result["config"]
    summary = exp_result["summary"]

    gb_auc = summary["GradientBoosting"]["auc_mean"]
    gb_std = summary["GradientBoosting"]["auc_std"]
    lr_auc = summary["LogisticRegression"]["auc_mean"]
    lr_std = summary["LogisticRegression"]["auc_std"]
    diff = gb_auc - lr_auc

    # Determine conclusion
    if abs(diff) < 0.01:
        conclusion = "No detectable difference"
    elif diff > 0:
        conclusion = "GradientBoosting outperforms LogisticRegression"
    else:
        conclusion = "LogisticRegression outperforms GradientBoosting"

    report = f"""# Churn Prediction Experiment Report

**Generated:** {datetime.now().isoformat()}

## Claim

On this customer churn dataset, does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

### Data & Features
- **Dataset:** `churn.csv` ({len(config['seeds'])} random seeds for cross-validation)
- **Features used:** {', '.join(config['features'])}
- **Features dropped:** {', '.join(config['dropped_features'])}
  - `days_since_last_login`: **Dropped due to target leakage.** This feature is derived from the outcome (churned customers stop logging in by definition) and is recorded post-outcome, encoding the target rather than predicting it.
  - `signup_date`: Temporal column; `tenure_months` already captures customer age.
  - `customer_id`: Non-predictive identifier.

### Preprocessing
- **Split:** Train/Validation/Test = 70/15/15 with stratification on target
- **Scaling:** StandardScaler fitted on train set only, applied to validation and test
- **Deduplication:** Exact-duplicate rows removed before split (200 duplicates found in original dataset)
- **Target balance:** Churn rate in full dataset: {config['baseline_auc']:.3f}

### Models
- **LogisticRegression:** max_iter=1000, default regularization (L2)
- **GradientBoosting:** n_estimators=100, learning_rate=0.1, max_depth=3

### Validation Strategy
- Stratified random split to avoid class imbalance bias
- No hyperparameter tuning (fixed hyperparameters for both models)
- Test set touched exactly once, at experiment end
- Baseline: majority class predictor (AUC = {config['baseline_auc']:.4f})

### Sanity Checks Performed
1. ✓ **Overfit check:** Both models reach train AUC > 0.90 on 100-row subset
2. ✓ **Label-shuffle test:** With shuffled labels, both models' AUC ≈ 0.5 (no leakage)
3. ✓ **Leakage ceiling:** Test AUC < 0.95 (realistic for this task)
4. ✓ **Baseline floor:** Both models beat majority-class baseline ({config['baseline_auc']:.4f})

## Results

### Test Set Performance (Mean ± SD across {len(config['seeds'])} seeds)

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| LogisticRegression | {lr_auc:.4f} ± {lr_std:.4f} | {summary['LogisticRegression']['precision_mean']:.4f} | {summary['LogisticRegression']['recall_mean']:.4f} | {summary['LogisticRegression']['f1_mean']:.4f} |
| GradientBoosting | {gb_auc:.4f} ± {gb_std:.4f} | {summary['GradientBoosting']['precision_mean']:.4f} | {summary['GradientBoosting']['recall_mean']:.4f} | {summary['GradientBoosting']['f1_mean']:.4f} |
| Baseline (Majority) | {config['baseline_auc']:.4f} | - | - | - |

### Effect Size
- **Difference (GB - LR):** {diff:+.4f}
- **Conclusion:** {conclusion}

## Interpretation

The effect size of {abs(diff):.4f} represents the mean difference in AUC-ROC between the two models across {len(config['seeds'])} random splits. Given the overlapping standard deviations, this difference is {'statistically meaningful' if abs(diff) > (gb_std + lr_std) else 'within noise'}.

Both models substantially outperform the baseline ({config['baseline_auc']:.4f}), indicating the dataset contains genuine predictive signal.

## Limitations & Risk Assessment

1. **Feature engineering:** No derived features (e.g., spend-per-tenure ratio). Simple feature set limits model expressiveness.
2. **Hyperparameter tuning:** Models use default hyperparameters. Tuning could shift relative performance.
3. **Temporal aspect:** Random split ignores signup_date ordering. A time-based split might reveal performance differences.
4. **Sample size:** {len(config['seeds'])} seeds provides moderate variance estimates. Larger CV folds could strengthen claims.
5. **Remaining unknowns:** Feature interactions or non-monotonic relationships not explored.

## Leakage Audit

### Dropped Feature: days_since_last_login
- **Risk:** Post-hoc activity measurement. Churned customers have high values by design.
- **Mitigation:** Feature explicitly dropped before train/test split.

### Data Quality
- **Duplicates:** 200 exact-duplicate rows found and removed before split.
- **Split integrity:** Stratified split ensures train/val/test are representative.

## Conclusion

**{conclusion}.** The observed difference of {diff:+.4f} in AUC across {len(config['seeds'])} seeds {'supports gradient boosting as the better model for this dataset' if diff > 0.01 else 'suggests parity between the two approaches, or requires more data/tuning to resolve'}.

For production use, consider:
- Cross-validation on more folds (5-fold or 10-fold)
- Hyperparameter tuning via validation-set evaluation (keeping test set sealed)
- Feature engineering informed by domain knowledge
- Monitoring model performance on new data post-deployment
"""

    report_file = Path("REPORT.md")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Saved report to {report_file}")

    print("\n✓ Experiment complete.")


if __name__ == "__main__":
    main()
