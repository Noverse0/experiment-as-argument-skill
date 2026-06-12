"""Customer churn prediction experiment: LogisticRegression vs GradientBoosting.

Design:
  Claim: GradientBoosting outperforms LogisticRegression at predicting churn.
  Variable: Model type (LR vs GB). All else held constant: same split, preprocessing, metrics.
  Data contact: Train/test split 80/20 stratified by churn. Preprocessing fit on train only.
  Leak surface:
    - account_status (leaked from target; dropped)
    - signup_date (temporal; dropped for this analysis)
    - Exact duplicates (detected and reported)
"""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path: str) -> pd.DataFrame:
    """Load churn dataset."""
    return pd.read_csv(csv_path)


def detect_duplicates(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Detect exact duplicates in the dataset. Return count and deduplicated df."""
    n_before = len(df)
    df_dedup = df.drop_duplicates(keep='first')
    n_after = len(df_dedup)
    n_dups = n_before - n_after
    return n_dups, df_dedup


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data: drop leaked/unused features, keep predictive ones.

    Drops:
      - customer_id (identifier, not predictive)
      - account_status (leaked from target: "closed" iff churned==1)
      - signup_date (temporal; not used for this cross-sectional analysis)

    Keeps:
      - tenure_months (likely predictive: positive feature)
      - monthly_spend (likely predictive: negative feature)
      - support_tickets (likely predictive: positive feature)
      - churned (target)
    """
    return df[['tenure_months', 'monthly_spend', 'support_tickets', 'churned']].copy()


def run_single_seed(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """Run one experiment iteration with a single seed. Return metrics dict."""

    X = df[['tenure_months', 'monthly_spend', 'support_tickets']]
    y = df['churned']

    # Stratified split preserves class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Fit preprocessing on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics = {'seed': seed}

    # LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    metrics['lr_auc'] = roc_auc_score(y_test, y_pred_proba_lr)
    metrics['lr_accuracy'] = (y_pred_lr == y_test).mean()
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
    metrics['lr_pr_auc'] = auc(recall_lr, precision_lr)

    # GradientBoosting
    gb = GradientBoostingClassifier(random_state=seed, n_iter_no_change=10)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

    metrics['gb_auc'] = roc_auc_score(y_test, y_pred_proba_gb)
    metrics['gb_accuracy'] = (y_pred_gb == y_test).mean()
    precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_pred_proba_gb)
    metrics['gb_pr_auc'] = auc(recall_gb, precision_gb)

    # Class balance in test
    metrics['test_churn_rate'] = y_test.mean()

    return metrics


def run_experiment(
    csv_path: str,
    n_seeds: int = 5,
    output_dir: str = "results",
) -> Dict:
    """Run full experiment with multiple seeds. Write results to JSON."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    n_dups, df_clean = detect_duplicates(df)

    print(f"Dataset: {len(df)} rows ({n_dups} duplicates detected, removed)")

    df_processed = preprocess_data(df_clean)
    print(f"Class balance: {df_processed['churned'].mean():.1%} churn rate")

    all_metrics = []
    for seed in range(n_seeds):
        metrics = run_single_seed(df_processed, seed=seed)
        all_metrics.append(metrics)
        print(f"  Seed {seed}: LR AUC={metrics['lr_auc']:.4f}, GB AUC={metrics['gb_auc']:.4f}")

    # Aggregate results
    df_results = pd.DataFrame(all_metrics)

    lr_auc = df_results['lr_auc'].values
    gb_auc = df_results['gb_auc'].values

    result = {
        'n_seeds': n_seeds,
        'n_duplicates': int(n_dups),
        'models': {
            'LogisticRegression': {
                'auc': {
                    'mean': float(lr_auc.mean()),
                    'std': float(lr_auc.std()),
                    'min': float(lr_auc.min()),
                    'max': float(lr_auc.max()),
                },
                'accuracy': {
                    'mean': float(df_results['lr_accuracy'].mean()),
                    'std': float(df_results['lr_accuracy'].std()),
                },
                'pr_auc': {
                    'mean': float(df_results['lr_pr_auc'].mean()),
                    'std': float(df_results['lr_pr_auc'].std()),
                },
            },
            'GradientBoosting': {
                'auc': {
                    'mean': float(gb_auc.mean()),
                    'std': float(gb_auc.std()),
                    'min': float(gb_auc.min()),
                    'max': float(gb_auc.max()),
                },
                'accuracy': {
                    'mean': float(df_results['gb_accuracy'].mean()),
                    'std': float(df_results['gb_accuracy'].std()),
                },
                'pr_auc': {
                    'mean': float(df_results['gb_pr_auc'].mean()),
                    'std': float(df_results['gb_pr_auc'].std()),
                },
            },
        },
    }

    # Compute effect size and verdict
    mean_diff = gb_auc.mean() - lr_auc.mean()
    pooled_std = np.sqrt((lr_auc.std()**2 + gb_auc.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    result['effect_size'] = {
        'mean_diff_auc': float(mean_diff),
        'cohens_d': float(cohens_d),
        'overlapping': bool(
            lr_auc.max() >= gb_auc.min() and gb_auc.max() >= lr_auc.min()
        ),
    }

    # Write results JSON
    results_path = Path(output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Write detailed metrics CSV
    df_results.to_csv(Path(output_dir) / "metrics_by_seed.csv", index=False)

    return result
