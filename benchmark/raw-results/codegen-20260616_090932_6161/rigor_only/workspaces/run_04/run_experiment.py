#!/usr/bin/env python3
"""Entrypoint: Run the full experiment and write results."""
import json
import sys
import os
from datetime import datetime
from pathlib import Path

from src.experiment import run_experiment


def write_results(result: dict, output_dir: str = 'results'):
    """Write machine-readable metrics and human-readable report."""
    Path(output_dir).mkdir(exist_ok=True)

    # Machine-readable: JSON.
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote metrics to {metrics_file}")

    # Human-readable: Markdown report.
    report_file = 'REPORT.md'
    summary = result['summary']
    config = result['config']

    lr_auc_mean = summary['LogisticRegression']['auc']['mean']
    lr_auc_std = summary['LogisticRegression']['auc']['std']
    gb_auc_mean = summary['GradientBoostingClassifier']['auc']['mean']
    gb_auc_std = summary['GradientBoostingClassifier']['auc']['std']
    gap = gb_auc_mean - lr_auc_mean

    conclusion = (
        "**Gradient Boosting OUTPERFORMS Logistic Regression**"
        if gap > 0
        else "**Gradient Boosting UNDERPERFORMS Logistic Regression**"
        if gap < 0
        else "**No detectable difference**"
    )

    report = f"""# Churn Prediction: LogisticRegression vs GradientBoostingClassifier

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Conclusion

{conclusion}

- LogisticRegression AUC: `{lr_auc_mean:.4f} ± {lr_auc_std:.4f}` (n={config['n_seeds']})
- GradientBoostingClassifier AUC: `{gb_auc_mean:.4f} ± {gb_auc_std:.4f}` (n={config['n_seeds']})
- Gap: `{gap:.4f}`

## Methodology

### Data
- **Source:** {config['data_path']}
- **Split:** {config['train_frac']:.0%} train / {1-config['train_frac']:.0%} test
- **Split method:** {config['split_method']}
- **Duplicates:** Deduplicated before splitting to prevent contamination

### Features
- **Included:** {', '.join(config['features'])}
- **Removed:** {'; '.join(config['removed_features'])}

### Preprocessing
1. Deduplicate exact rows
2. Remove target leakage (days_since_last_login encodes the outcome post-facto)
3. Engineer days_since_signup from temporal column
4. Standard scale features (fit on train, apply to test)
5. Time-based split (respect signup_date order)

### Models
- **LogisticRegression:** Default sklearn parameters, fitted on scaled features
- **GradientBoostingClassifier:** Default sklearn parameters, fitted on scaled features

### Experiment Design
- **Seeds:** {config['n_seeds']}
- **Metrics:** ROC-AUC (primary), F1, precision, recall, accuracy
- **Sanity checks:**
  - Baseline floor (majority class prediction)
  - Label shuffle test (verify no spurious leakage)
  - One check per seed ensures controls are repeatable

## Limitations & Risk

### Potential Leakage Surface
- ✓ **days_since_last_login** removed (encodes outcome timing)
- ✓ **Deduplication** prevents train-test contamination
- ✓ **Time-based split** respects signup_date order
- ⚠ **days_since_signup** derived from signup_date (safe, but could drift over time in production)

### Generalization
- Dataset is synthetic; real churn may have different feature relationships
- 4200 rows is small; variance estimates may be noisy
- Hyperparameters not tuned; this is a fair-baseline comparison, not an optimization race

## Raw Metrics

See `results/metrics.json` for per-seed results.
"""

    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Wrote report to {report_file}")


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    print(f"Output directory: {output_dir}")

    result = run_experiment(output_dir=output_dir, n_seeds=5)
    write_results(result, output_dir=output_dir)

    print("\n✓ Experiment complete. See REPORT.md for conclusions.")
