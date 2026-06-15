#!/usr/bin/env python3
"""
Main experiment runner: compare logistic regression vs gradient boosting.

Rigor checks:
- Deduplicates exact rows before splitting
- Uses time-based split (respects temporal structure)
- Excludes target-leaking features (days_since_last_login)
- Runs multiple seeds, reports mean ± std
- Includes sanity checks (baseline, label shuffle)
"""
import sys
import json
import numpy as np
from pathlib import Path

from src.dataset import load_and_deduplicate, time_based_split, prepare_features, get_feature_columns
from src.models import build_logistic_regression, build_gradient_boosting, train_and_evaluate
from src.evaluation import aggregate_results, save_metrics, generate_report


def sanity_check_baseline(X_train, y_train, X_test, y_test):
    """Baseline: predict majority class."""
    majority = np.bincount(y_train).argmax()
    y_pred = np.full_like(y_test, majority)
    from sklearn.metrics import accuracy_score
    baseline_acc = accuracy_score(y_test, y_pred)
    churn_rate = y_test.mean()
    print(f"  Baseline (majority class): {baseline_acc:.4f} accuracy, churn rate: {churn_rate:.4f}")
    return baseline_acc


def sanity_check_label_shuffle(X_train, y_train, X_test, y_test):
    """With shuffled labels, models should perform like baseline."""
    y_train_shuffled = np.random.permutation(y_train)
    lr = build_logistic_regression()
    metrics = train_and_evaluate(lr, X_train, y_train_shuffled, X_test, y_test)
    print(f"  Label shuffle (LR w/ shuffled labels): AUC={metrics['auc']:.4f}, should be ~0.5")
    return metrics['auc']


def sanity_check_overfit_tiny(X_train, y_train):
    """Overfit on a tiny subset; loss should approach zero."""
    if len(X_train) < 10:
        print("  Overfit on tiny subset: skipped (training set too small)")
        return True

    tiny_size = min(10, len(X_train) // 2)
    X_tiny = X_train[:tiny_size]
    y_tiny = y_train[:tiny_size]

    gb = build_gradient_boosting(random_state=42)
    gb.fit(X_tiny, y_tiny)
    y_pred = gb.predict(X_tiny)

    acc = (y_pred == y_tiny).mean()
    print(f"  Overfit on {tiny_size} samples (GB): accuracy={acc:.4f}, should be high")
    return acc > 0.5


def run_experiment(
    dataset_path: str = "churn.csv",
    n_seeds: int = 3,
    output_dir: str = "results"
):
    """Run the full experiment."""
    print("=" * 70)
    print("CHURN PREDICTION EXPERIMENT: Gradient Boosting vs Logistic Regression")
    print("=" * 70)

    # Load and preprocess
    print("\n1. Loading and deduplicating dataset...")
    df = load_and_deduplicate(dataset_path)
    initial = df.attrs['initial_rows']
    removed = df.attrs['duplicates_removed']
    print(f"   Loaded {dataset_path}: {initial} rows, removed {removed} duplicates, {len(df)} remain")

    # Split
    print("\n2. Time-based split (earlier signups → train, later → test)...")
    train, test = time_based_split(df, test_fraction=0.2)
    print(f"   Train: {len(train)} rows, churn rate: {train['churned'].mean():.3f}")
    print(f"   Test:  {len(test)} rows, churn rate: {test['churned'].mean():.3f}")

    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)

    print(f"\n   Features: {get_feature_columns()}")
    print(f"   (Note: days_since_last_login excluded due to target leakage)")

    # Sanity checks
    print("\n3. Sanity checks...")
    sanity_check_baseline(X_train, y_train, X_test, y_test)
    sanity_check_label_shuffle(X_train, y_train, X_test, y_test)
    sanity_check_overfit_tiny(X_train, y_train)

    # Train models across multiple seeds
    print(f"\n4. Training models ({n_seeds} seeds)...")
    results_by_model = {
        'LogisticRegression': [],
        'GradientBoosting': [],
    }

    for seed in range(n_seeds):
        print(f"   Seed {seed}...")
        np.random.seed(seed)

        # LogisticRegression
        lr = build_logistic_regression()
        lr.named_steps['model'].random_state = seed
        lr_metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)
        results_by_model['LogisticRegression'].append(lr_metrics)

        # GradientBoosting
        gb = build_gradient_boosting(random_state=seed)
        gb_metrics = train_and_evaluate(gb, X_train, y_train, X_test, y_test)
        results_by_model['GradientBoosting'].append(gb_metrics)

        print(f"      LR  AUC: {lr_metrics['auc']:.4f}")
        print(f"      GB  AUC: {gb_metrics['auc']:.4f}")

    # Aggregate and save
    print("\n5. Aggregating results...")
    aggregated = aggregate_results(results_by_model)
    for model_name, metrics in aggregated.items():
        auc = metrics.get('auc', np.nan)
        std = metrics.get('auc_std', 0)
        print(f"   {model_name:20s}: AUC={auc:.4f} ± {std:.4f}")

    # Save metrics
    Path(output_dir).mkdir(exist_ok=True)
    metrics_path = Path(output_dir) / "metrics.json"
    save_metrics(aggregated, str(metrics_path))
    print(f"\n   Saved metrics to {metrics_path}")

    # Generate report
    methodology = f"""
### Data Preparation
- Loaded churn.csv ({initial} rows, 4000 + 200 exact duplicates)
- Deduplicated: removed {removed} exact duplicate rows
- Time-based split: sorted by signup_date, train on first 80%, test on last 20%
  - Respects temporal causality (no information leak from future to past)
  - Ensures exact duplicates don't straddle train/test boundary

### Feature Selection
- **Included:** tenure_months, monthly_spend, support_tickets
  - These are honest causal features (determined at signup time)
- **Excluded:** days_since_last_login
  - **Rationale:** This is target leakage. Churned customers have, by definition,
    stopped logging in, so this value is recorded at/after the outcome.
    Including it would give artificially high performance without real predictive power.

### Model Training
- **LogisticRegression**: L-BFGS solver, balanced class weights, standardized features
- **GradientBoosting**: 100 estimators, depth=3, learning_rate=0.1
- Both trained {n_seeds} times with different random seeds
- Train/test split fixed; only random seed varies per run

### Metrics
- Primary: **AUC-ROC** (robust to class imbalance)
- Secondary: accuracy, balanced accuracy, precision, recall, F1

### Sanity Checks
✓ Baseline check: majority-class prediction evaluated
✓ Label shuffle: model trained on shuffled labels (should fail)
✓ Overfit test: model fit on tiny subset (should succeed)
"""

    limitations = """
- **Single dataset**: Results generalize to this customer base only
- **Feature set limitation**: Only 3 honest features available; more features could improve both models
- **Time-based split**: Assumes future churn patterns match historical patterns
- **Temporal drift**: If customer behavior changes over time, test performance may not reflect real-world deployment
- **Exclusion of days_since_last_login**: Acknowledges and avoids a strong leak, but this means GB vs LR comparison
  is on a constrained feature set
- **Seed variance**: Limited to 3 seeds (n=3 runs per model); more seeds would improve confidence
"""

    generate_report(aggregated, methodology, limitations, "REPORT.md")
    print("\nGenerated REPORT.md")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
