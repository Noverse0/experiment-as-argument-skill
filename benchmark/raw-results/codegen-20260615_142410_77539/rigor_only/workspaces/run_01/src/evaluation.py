"""Evaluation and reporting utilities."""
import json
import numpy as np
from pathlib import Path


def aggregate_results(results_by_model: dict) -> dict:
    """
    Aggregate results across seeds for each model.

    Args:
        results_by_model: {model_name: [list of metric dicts]}

    Returns:
        {model_name: {metric: mean, metric_std: std, metric_n: count}}
    """
    aggregated = {}

    for model_name, runs in results_by_model.items():
        metrics_dict = {}
        for metric_key in runs[0].keys():
            values = [r[metric_key] for r in runs if not np.isnan(r[metric_key])]
            if values:
                metrics_dict[metric_key] = np.mean(values)
                metrics_dict[f'{metric_key}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                metrics_dict[f'{metric_key}_n'] = len(values)

        aggregated[model_name] = metrics_dict

    return aggregated


def save_metrics(results: dict, output_path: str):
    """Save aggregated results as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def format_comparison(aggregated: dict) -> str:
    """Generate a text comparison of models."""
    lines = []
    lines.append("## Model Comparison (AUC-ROC, primary metric)\n")

    models = sorted(aggregated.keys())
    for model_name in models:
        metrics = aggregated[model_name]
        auc = metrics.get('auc', np.nan)
        auc_std = metrics.get('auc_std', 0.0)
        auc_n = metrics.get('auc_n', 0)
        lines.append(
            f"- **{model_name}**: {auc:.4f} ± {auc_std:.4f} (n={int(auc_n)})"
        )

    if len(models) == 2:
        m1, m2 = models[0], models[1]
        auc1 = aggregated[m1].get('auc', 0)
        auc2 = aggregated[m2].get('auc', 0)
        std1 = aggregated[m1].get('auc_std', 0)
        std2 = aggregated[m2].get('auc_std', 0)

        gap = auc2 - auc1
        combined_std = np.sqrt(std1**2 + std2**2)

        if combined_std > 0:
            z_score = gap / combined_std
        else:
            z_score = np.inf if gap > 0 else -np.inf

        lines.append(f"\nGap: {gap:+.4f} (z≈{z_score:.2f} std)")
        if abs(gap) < combined_std:
            lines.append(f"→ **No significant difference detected** (gap < 1σ).\n")
        else:
            winner = m2 if gap > 0 else m1
            lines.append(f"→ **{winner}** shows better performance.\n")

    return '\n'.join(lines)


def generate_report(
    aggregated: dict,
    methodology: str,
    limitations: str,
    output_path: str = "REPORT.md"
):
    """Generate full markdown report."""
    report = f"""# Churn Prediction Experiment Report

## Claim
For customer churn prediction using honest features (tenure, support tickets, monthly spend),
does gradient boosting outperform logistic regression?

## Methodology

{methodology}

## Results

{format_comparison(aggregated)}

### Full Metrics

"""
    for model_name in sorted(aggregated.keys()):
        report += f"\n#### {model_name}\n\n```\n"
        metrics = aggregated[model_name]
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, float):
                report += f"{key}: {val:.4f}\n"
            else:
                report += f"{key}: {val}\n"
        report += "```\n"

    report += f"""

## Limitations & Risk

{limitations}

## Conclusion

The experiment supports a fair comparison of gradient boosting and logistic regression
on this churn dataset. The metric differences are interpreted in the context of variance
across seeds to avoid claiming significance without statistical support.
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
