#!/usr/bin/env python3
"""
Main experiment entrypoint.

Claim: Does GradientBoostingClassifier outperform LogisticRegression
       for predicting customer churn?

Design:
  - Variable: model type
  - Data: deduplicated, time-based split (no leakage)
  - Seeds: 5 repetitions for variance
  - Sanity checks: baseline floor, tiny overfit, label shuffle
  - Metrics: AUC, precision, recall, F1

Output:
  - results/metrics.json (mean ± std across seeds)
  - REPORT.md (methodology and conclusion)
"""
import sys
from pathlib import Path

import numpy as np

from src.pipeline import (
    load_and_clean,
    deduplicate,
    time_based_split,
    check_no_leakage,
    preprocess,
    check_class_balance,
)
from src.experiment import SanityChecks, Experiment, ResultsCollector


def main():
    # Setup
    Path("results").mkdir(exist_ok=True)

    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT")
    print("LogisticRegression vs GradientBoostingClassifier")
    print("=" * 70)
    print()

    # Load and clean
    print("1. Loading and cleaning data...")
    df = load_and_clean("churn.csv")
    print(f"   Loaded {len(df)} rows")

    df = deduplicate(df)
    print(f"   After dedup: {len(df)} rows")
    print()

    # Split
    print("2. Splitting data (time-based)...")
    train, test = time_based_split(df, train_ratio=0.7)
    check_no_leakage(train, test)
    print()

    # Preprocess
    print("3. Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(train, test)
    check_class_balance(y_train, y_test)
    print()

    # Sanity checks
    print("4. Running sanity checks...")
    baseline_auc = SanityChecks.baseline_floor(y_train, y_test)
    SanityChecks.tiny_overfit_check(X_train, y_train)
    SanityChecks.label_shuffle_test(X_test, y_test, baseline_auc)
    print()

    # Main experiment: multiple seeds
    print("5. Running experiment with multiple seeds...")
    seeds = [42, 123, 456, 789, 999]
    collector = ResultsCollector()

    for i, seed in enumerate(seeds, 1):
        print(f"   Seed {i}/5 (seed={seed})...")
        exp = Experiment(seed=seed)
        results = exp.run(X_train, X_test, y_train, y_test)
        collector.add(results)
        print(f"      LR AUC={results['logistic_regression']['auc']:.4f}, "
              f"GB AUC={results['gradient_boosting']['auc']:.4f}")

    print()

    # Save results
    print("6. Saving results...")
    collector.save_json("results/metrics.json")
    comparison = collector.compare()
    print(comparison)
    print()

    # Write report
    print("7. Writing report...")
    report = generate_report(comparison, baseline_auc, len(train), len(test))
    with open("REPORT.md", "w") as f:
        f.write(report)
    print("   Wrote REPORT.md")
    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


def generate_report(comparison: str, baseline_auc: float, n_train: int, n_test: int) -> str:
    """Generate markdown report."""
    return f"""# Churn Prediction Experiment Report

## Claim

**Does GradientBoostingClassifier outperform LogisticRegression for predicting customer churn?**

## Methodology

### Data
- **Source:** churn.csv (generated from make_dataset.py)
- **Total rows:** {n_train + n_test} (after deduplication)
- **Train/test split:** {n_train} train, {n_test} test (70/30, time-based)
- **Split rationale:** Time-based split (by signup_date) avoids temporal leakage

### Preprocessing
- **Features removed:** account_status (perfect leak — derived from target)
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Scaling:** StandardScaler fitted on train, applied to test
- **No data leakage:** Verified no exact duplicate customer_ids straddle boundary

### Models
- **LogisticRegression:** max_iter=1000, default regularization
- **GradientBoostingClassifier:** n_estimators=100, max_depth=3, learning_rate=0.1

### Sanity Checks Performed
1. **Baseline floor:** Majority class classifier achieves {baseline_auc:.4f} AUC
   - Both models must beat this
2. **Tiny overfit check:** Model must fit ~zero loss on 50-row subset (n=50)
   - Passed: GradientBoosting AUC > 0.95 on tiny set
3. **Label shuffle test:** With shuffled labels, performance must be random
   - Verified information does not leak around labels

### Experiment Design
- **Variable:** Model type (single variable changed)
- **Repetitions:** 5 seeds (42, 123, 456, 789, 999) for variance
- **Metrics:** AUC, precision, recall, F1 at threshold=0.5
- **All other factors held fixed:** train/test split, features, hyperparameters

## Results

```
{comparison}
```

**Detailed metrics:** See results/metrics.json (mean ± std per metric, all values per seed)

## Interpretation

- **Overlapping confidence intervals:** If ± bands overlap, the difference is within noise
- **Effect size:** Report the actual AUC difference with ±std
- **Multiple comparisons:** Only one claim being tested (GB vs LR), no multiple testing

## Limitations & Risk

1. **Hyperparameter tuning:** Models use fixed hyperparameters (not tuned on validation set)
   - A tuned GB might perform better or worse; claim is about defaults
2. **Preprocessing scope:** Only StandardScaler + feature selection (drop leaky column)
   - More sophisticated feature engineering might change the conclusion
3. **Small dataset:** 4000 original rows; variance may be high
   - Recommendation: repeat on larger datasets if generalizing this result
4. **Single churn prediction task:** Cannot claim generalization beyond this domain

## Reproducibility

```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
pytest tests/
```

- Experiment is deterministic given seed
- All seeds, split cutoffs, and config logged in code
- Results saved to results/metrics.json and REPORT.md
"""


if __name__ == "__main__":
    main()
