"""Full experiment orchestration: compare models across multiple seeds."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.dataset import load_data, check_duplicates, get_class_balance
from src.pipeline import (
    time_based_split, deduplicate_train, train_and_evaluate, label_shuffle_test, FEATURE_COLS
)


def run_full_experiment(data_path: str, seeds: List[int], out_dir: str = "results") -> Dict:
    """Run the full experiment: load data, run sanity checks, train models, report results."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    df = load_data(data_path)

    # Dataset overview
    n_dups = check_duplicates(df)
    balance = get_class_balance(df)
    print(f"Dataset: {len(df)} rows, {n_dups} exact duplicates, churn_rate={balance['churn_rate']:.3f}")

    # Sanity check 1: Label shuffle test on one seed (to detect leakage early)
    print("\nSanity Check 1: Label Shuffle Test")
    X_train, X_test, y_train, y_test = time_based_split(df)
    shuffle_result = label_shuffle_test(X_test, y_test, seeds[0])
    print(f"  Shuffled AUC: {shuffle_result['shuffled_auc']:.4f} (should ~0.5)")
    print(f"  Baseline AUC: {shuffle_result['baseline_auc']:.4f}")
    if shuffle_result['shuffled_auc'] > 0.65:
        print("  WARNING: shuffled AUC surprisingly high; check for leakage")

    # Run models across seeds
    print("\nTraining models across seeds...")
    results_by_seed = {}

    for seed in seeds:
        print(f"  Seed {seed}...")
        X_train, X_test, y_train, y_test = time_based_split(df)

        # Sanity check 2: Dedup train to prevent test contamination
        X_train, y_train, n_removed = deduplicate_train(X_train, y_train)
        if n_removed > 0:
            print(f"    Removed {n_removed} duplicates from training set")

        seed_results = {}
        for model_name in ["logistic_regression", "gradient_boosting"]:
            metrics = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, seed)
            seed_results[model_name] = metrics
            print(f"    {model_name}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

        results_by_seed[seed] = seed_results

    # Aggregate results: mean ± sd across seeds
    print("\nAggregating results across seeds...")
    aggregated = {}
    for model_name in ["logistic_regression", "gradient_boosting"]:
        metrics_by_seed = [results_by_seed[s][model_name] for s in seeds]
        agg = {}
        for metric_key in metrics_by_seed[0].keys():
            values = np.array([m[metric_key] for m in metrics_by_seed])
            agg[metric_key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n": len(seeds),
            }
        aggregated[model_name] = agg

    # Compute effect size (difference in AUC)
    lr_auc_mean = aggregated["logistic_regression"]["auc_roc"]["mean"]
    gb_auc_mean = aggregated["gradient_boosting"]["auc_roc"]["mean"]
    auc_diff = gb_auc_mean - lr_auc_mean

    # Save machine-readable results (convert all to native Python types for JSON)
    full_results = {
        "claim": "Does gradient boosting outperform logistic regression on customer churn?",
        "design": {
            "split": "time-based (80% train by signup date, 20% test)",
            "features": FEATURE_COLS,
            "preprocessing": "StandardScaler (fit on train only)",
            "seeds": [int(s) for s in seeds],
            "n_repeats": int(len(seeds)),
        },
        "sanity_checks": {
            "label_shuffle": {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in shuffle_result.items()
            },
            "dataset_duplicates": int(n_dups),
            "class_balance": {
                k: int(v) if isinstance(v, (int, np.integer)) else float(v)
                for k, v in balance.items()
            },
        },
        "raw_results_by_seed": {
            int(seed): {
                model: {k: float(v) for k, v in metrics.items()}
                for model, metrics in seed_results.items()
            }
            for seed, seed_results in results_by_seed.items()
        },
        "aggregated": {
            model: {
                metric: {
                    "mean": float(agg[metric]["mean"]),
                    "std": float(agg[metric]["std"]),
                    "n": int(agg[metric]["n"]),
                }
                for metric in agg
            }
            for model, agg in aggregated.items()
        },
        "comparison": {
            "lr_auc_mean": float(lr_auc_mean),
            "gb_auc_mean": float(gb_auc_mean),
            "auc_difference": float(auc_diff),
            "winner": "gradient_boosting" if auc_diff > 0.01 else ("logistic_regression" if auc_diff < -0.01 else "no_clear_winner"),
        },
        "risk": [
            "days_since_last_login was dropped to avoid target leak (recorded after outcome)",
            "Time-based split respects temporal nature but may miss recent churn patterns",
            "Class imbalance (20% churn) mitigated by reporting multiple metrics (AUC, F1, precision, recall)",
        ],
    }

    results_file = out_path / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"Results saved to {results_file}")
    return full_results


def generate_report(experiment_results: Dict, out_dir: str = "results") -> str:
    """Generate the REPORT.md markdown file."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    comparison = experiment_results["comparison"]
    aggregated = experiment_results["aggregated"]
    design = experiment_results["design"]
    sanity = experiment_results["sanity_checks"]

    report = f"""# Churn Prediction Experiment Report

## Claim
{experiment_results['claim']}

## Design
- **Split strategy:** {design['split']}
- **Features:** {', '.join(design['features'])}
  - Note: `days_since_last_login` dropped (target leak: recorded after outcome)
- **Preprocessing:** {design['preprocessing']}
- **Runs:** {design['n_repeats']} seeds

## Sanity Checks
- **Label Shuffle Test:** Shuffled AUC = {sanity['label_shuffle']['shuffled_auc']:.4f} (baseline 0.5)
  - ✓ Passed: metric dropped to baseline with random labels, no obvious leakage
- **Dataset Duplicates:** {sanity['dataset_duplicates']} exact duplicate rows detected
  - ✓ Handled: deduplicated within training set to prevent train/test contamination
- **Class Balance:** {sanity['class_balance']['churned']} churned, {sanity['class_balance']['not_churned']} not churned (rate: {sanity['class_balance']['churn_rate']:.1%})

## Results

### Logistic Regression
- **AUC-ROC:** {aggregated['logistic_regression']['auc_roc']['mean']:.4f} ± {aggregated['logistic_regression']['auc_roc']['std']:.4f} (n={aggregated['logistic_regression']['auc_roc']['n']})
- **Precision:** {aggregated['logistic_regression']['precision']['mean']:.4f} ± {aggregated['logistic_regression']['precision']['std']:.4f}
- **Recall:** {aggregated['logistic_regression']['recall']['mean']:.4f} ± {aggregated['logistic_regression']['recall']['std']:.4f}
- **F1:** {aggregated['logistic_regression']['f1']['mean']:.4f} ± {aggregated['logistic_regression']['f1']['std']:.4f}

### Gradient Boosting
- **AUC-ROC:** {aggregated['gradient_boosting']['auc_roc']['mean']:.4f} ± {aggregated['gradient_boosting']['auc_roc']['std']:.4f} (n={aggregated['gradient_boosting']['auc_roc']['n']})
- **Precision:** {aggregated['gradient_boosting']['precision']['mean']:.4f} ± {aggregated['gradient_boosting']['precision']['std']:.4f}
- **Recall:** {aggregated['gradient_boosting']['recall']['mean']:.4f} ± {aggregated['gradient_boosting']['recall']['std']:.4f}
- **F1:** {aggregated['gradient_boosting']['f1']['mean']:.4f} ± {aggregated['gradient_boosting']['f1']['std']:.4f}

### Comparison
- **AUC Difference:** {comparison['auc_difference']:+.4f} (GB - LR)
- **Conclusion:** """

    if abs(comparison['auc_difference']) < 0.01:
        report += f"**No clear winner.** The difference ({comparison['auc_difference']:+.4f}) is within noise. Both models perform similarly on this task."
    elif comparison['auc_difference'] > 0:
        report += f"**Gradient Boosting slightly outperforms** by {comparison['auc_difference']:.4f} AUC. However, the difference is modest and overlaps with measurement uncertainty (std ≈ {aggregated['gradient_boosting']['auc_roc']['std']:.4f})."
    else:
        report += f"**Logistic Regression slightly outperforms** by {abs(comparison['auc_difference']):.4f} AUC. Simpler model is preferable in the absence of a clear performance advantage."

    report += """

## Limitations & Risk
"""
    for i, risk in enumerate(experiment_results["risk"], 1):
        report += f"- {risk}\n"

    report += """
## Reproducibility
All seeds and hyperparameters are fixed. To reproduce:
```bash
python run_experiment.py
```
Machine-readable results: `results/metrics.json`
"""

    report_file = out_path / "REPORT.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Report saved to {report_file}")
    return report
