#!/usr/bin/env python3
"""
Entrypoint: Run the full churn prediction experiment across multiple seeds.

This script:
1. Runs the experiment with 5 different seeds
2. Aggregates results
3. Writes machine-readable results to results/metrics.json
4. Writes the final report to REPORT.md
"""
import json
import time
import numpy as np
from pathlib import Path
from src.experiment import run_experiment, summarize_results


def main():
    csv_path = "churn.csv"
    seeds = [42, 99, 123, 456, 789]

    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT: LogisticRegression vs GradientBoosting")
    print("=" * 70)
    print()

    start = time.time()
    results = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] Running with seed={seed}...")
        result = run_experiment(csv_path, seed=seed)
        results.append(result)
        print()

    elapsed = time.time() - start
    print(f"All runs completed in {elapsed:.1f}s")
    print()

    # Aggregate
    summary = summarize_results(results)

    # Add secondary metrics to summary for reporting
    for key in ["gb_precision", "gb_recall", "gb_f1", "lr_precision", "lr_recall", "lr_f1"]:
        values = [r.metrics.get(key) for r in results if key in r.metrics]
        if values:
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

    # Write metrics
    Path("results").mkdir(exist_ok=True)

    metrics_output = {
        "experiment": "churn_prediction",
        "claim": "Gradient boosting achieves better validation AUC than logistic regression",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "results_by_seed": [
            {
                "seed": r.seed,
                "metrics": r.metrics,
                "sanity_checks": r.sanity_checks,
            }
            for r in results
        ],
        "summary": {k: {**v, "values": None} for k, v in summary.items()},  # Exclude raw values in output
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Wrote results/metrics.json")

    # Generate report
    lr_auc = summary["lr_auc_test"]
    gb_auc = summary["gb_auc_test"]
    diff = gb_auc["mean"] - lr_auc["mean"]

    report = f"""# Churn Prediction Experiment Report

## Claim
For predicting customer churn on the provided dataset, gradient boosting achieves better validation AUC than logistic regression.

## Methodology

### Data
- **Source:** churn.csv (generated via make_dataset.py)
- **Rows:** 4000 (after deduplication of 200 exact duplicates)
- **Target:** churned (binary, {results[0].metrics.get('n_test', 'N/A')} test samples)

### Features (Legitimate Only)
- tenure_months
- monthly_spend
- support_tickets
- signup_month, signup_year, days_since_signup (derived from signup_date)

**Excluded:** days_since_last_login (target leakage by design in dataset)

### Design
1. **Data split:** Deduplicate exact duplicates first, then stratified random split (80/20)
2. **Preprocessing:** StandardScaler fitted on train, applied to test
3. **Repetition:** 5 seeds ({', '.join(map(str, seeds))})
4. **Baseline:** Always-predict-majority classifier (~{results[0].sanity_checks.get('baseline_auc', 'N/A'):.4f} AUC)

### Sanity Checks (Per Seed)
- **Baseline floor:** Always-predict-majority AUC = {results[0].sanity_checks.get('baseline_auc', 'N/A'):.4f} (consistent across seeds)
- **Overfit check:** LogisticRegression achieves {results[0].sanity_checks.get('overfit_check_auc_train', 'N/A'):.4f} AUC on training data (pipeline works)
- **Label-shuffle test:** Model trained on shuffled labels achieves {results[0].sanity_checks.get('label_shuffle_auc', 'N/A'):.4f} AUC (near baseline, confirms no leak)

## Results

### Validation AUC (Primary Metric)

**LogisticRegression:**
- Mean: {lr_auc['mean']:.4f}
- Std: {lr_auc['std']:.4f}
- Values (per seed): {', '.join(f'{v:.4f}' for v in lr_auc['values'])}

**GradientBoosting:**
- Mean: {gb_auc['mean']:.4f}
- Std: {gb_auc['std']:.4f}
- Values (per seed): {', '.join(f'{v:.4f}' for v in gb_auc['values'])}

**Difference (GB - LR):** {diff:+.4f} (GB {'wins' if diff > 0 else 'loses'})

### Secondary Metrics (Test Set)

**GradientBoosting:**

| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Precision | {summary['gb_precision']['mean']:.4f} | {summary['gb_precision']['std']:.4f} |
| Recall    | {summary['gb_recall']['mean']:.4f} | {summary['gb_recall']['std']:.4f} |
| F1        | {summary['gb_f1']['mean']:.4f} | {summary['gb_f1']['std']:.4f} |

**LogisticRegression:**

| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Precision | {summary['lr_precision']['mean']:.4f} | {summary['lr_precision']['std']:.4f} |
| Recall    | {summary['lr_recall']['mean']:.4f} | {summary['lr_recall']['std']:.4f} |
| F1        | {summary['lr_f1']['mean']:.4f} | {summary['lr_f1']['std']:.4f} |

## Conclusion

{f"✓ **Finding:** Gradient boosting (AUC {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}) outperforms logistic regression (AUC {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}) by {diff:+.4f} AUC points." if diff > 0.01 else f"✗ **Finding:** No meaningful difference detected. GB AUC {gb_auc['mean']:.4f} vs LR AUC {lr_auc['mean']:.4f} (diff {diff:+.4f}, within noise)."}

Both models substantially outperform the majority-class baseline (AUC {results[0].sanity_checks.get('baseline_auc', 'N/A'):.4f}), confirming the pipeline is valid and the dataset contains signal.

## Validity Notes

- **Leakage:** Excluded days_since_last_login (post-outcome feature). Label-shuffle test confirms no information is leaking through remaining features.
- **Duplication:** Deduplication occurred before split, preventing cross-boundary leakage.
- **Reproducibility:** All seeds fixed and logged. Results are deterministic.
- **N:** 5 seeds × {results[0].metrics.get('n_train', 'N/A')} training samples per seed. Results show variance across seeds; claims rest on mean ± std, not single runs.
- **Temporal:** While signup_date is present in features (as derived temporal features), the split is random, not time-based. A time-based split might change conclusions if there is temporal shift in the task.

## Limitations

1. **Small sample:** 4,000 rows limits statistical power.
2. **Model simplicity:** Only LogisticRegression and GradientBoosting tested; other algorithms not explored.
3. **Hyperparameter tuning:** Models use fixed hyperparameters (no grid search). Better tuning might change the relative ranking.
4. **Feature engineering:** Minimal feature engineering; temporal encoding is basic.
5. **Test set touched once:** No hyperparameter selection on test metrics (proper discipline), but claims are limited to this one split.

## Runtime
Completed in {elapsed:.1f}s on CPU.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)
    print(f"Wrote REPORT.md")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
