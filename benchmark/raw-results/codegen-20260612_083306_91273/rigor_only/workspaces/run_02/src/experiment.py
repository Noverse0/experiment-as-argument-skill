"""
Churn prediction experiment: Gradient Boosting vs Logistic Regression.

Experiment design:
- Claim: GB outperforms LR on ROC-AUC for customer churn prediction
- Variable: model type (all else fixed)
- Data: train/test 80/20 stratified, deduplicated before split
- Leakage: account_status excluded (derived from target), signup_date converted to days
- Sanity: baseline floor, overfit check, label-shuffle test
- Repetition: 5 seeds per model, report mean ± std
"""
import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class RunResult(NamedTuple):
    seed: int
    model_name: str
    roc_auc_test: float
    precision_test: float
    recall_test: float
    f1_test: float
    baseline_auc: float
    n_train: int
    n_test: int
    churn_rate: float
    dedup_rows_removed: int


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, float, int]:
    """
    Load dataset, identify and remove duplicates.

    Returns:
      (df_dedup, target_rate, n_duplicates_removed)
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)

    # Check for exact duplicates on customer_id (all rows should be unique)
    df_dedup = df.drop_duplicates(subset=['customer_id'], keep='first')
    n_dedup = n_before - len(df_dedup)

    # Compute target rate for reporting
    target_rate = df_dedup['churned'].mean()

    logger.info(f"Loaded {n_before} rows, removed {n_dedup} duplicates, {len(df_dedup)} rows remain")
    logger.info(f"Churn rate: {target_rate:.4f}")

    return df_dedup, target_rate, n_dedup


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signup_date to days since signup (temporal leakage prevention).
    Exclude account_status (leaked from target).
    """
    df = df.copy()
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    max_date = df['signup_date'].max()
    df['days_since_signup'] = (max_date - df['signup_date']).dt.days

    # Drop account_status (leaked) and signup_date (raw temporal), keep days_since_signup
    df = df.drop(columns=['account_status', 'signup_date', 'customer_id'])
    return df


def run_sanity_checks(X_train, y_train, X_test, y_test, seed: int = 42) -> dict:
    """
    Run sanity checks: baseline, overfit, label-shuffle.

    Returns dict of check results.
    """
    checks = {}

    # 1. Baseline floor: majority class
    baseline_pred = np.ones(len(y_test)) * (y_train.mean() > 0.5)
    baseline_auc = roc_auc_score(y_test, baseline_pred)
    checks['baseline_auc'] = float(baseline_auc)
    logger.info(f"Baseline (majority class) AUC: {baseline_auc:.4f}")

    # 2. Overfit on small subset
    n_small = max(20, min(100, len(X_train) // 10))
    # Ensure both classes are present in the small subset
    y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
    idx_small = np.where(y_train_array[:n_small])[0]
    if len(idx_small) == 0:  # If no positive examples, find some
        idx_small = np.where(y_train_array)[0][:n_small//2]
    idx_negative = np.where(~y_train_array)[0][:n_small//2]
    idx_combined = np.concatenate([idx_small, idx_negative])[:n_small]

    if len(idx_combined) >= 4:  # Need at least 4 samples for meaningful evaluation
        X_small = X_train.iloc[idx_combined]
        y_small = y_train.iloc[idx_combined]
        lr_overfit = LogisticRegression(max_iter=1000, random_state=seed)
        lr_overfit.fit(X_small, y_small)
        train_auc = roc_auc_score(y_small, lr_overfit.predict_proba(X_small)[:, 1])
        checks['overfit_auc'] = float(train_auc)
        logger.info(f"Overfit check (LR on {len(idx_combined)} rows): AUC = {train_auc:.4f}")
    else:
        checks['overfit_auc'] = 0.5  # Fallback for very small datasets
        logger.info(f"Overfit check skipped (dataset too small)")

    # 3. Label shuffle test
    y_train_shuffled = y_train.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(y_train_shuffled.values)
    lr_shuffle = LogisticRegression(max_iter=1000, random_state=seed)
    lr_shuffle.fit(X_train, y_train_shuffled)
    shuffle_auc = roc_auc_score(y_test, lr_shuffle.predict_proba(X_test)[:, 1])
    checks['shuffle_auc'] = float(shuffle_auc)
    logger.info(f"Label-shuffle test (LR with shuffled labels): AUC = {shuffle_auc:.4f}")

    if shuffle_auc > baseline_auc + 0.05:
        logger.warning(f"⚠️  Label-shuffle AUC {shuffle_auc:.4f} is not near baseline {baseline_auc:.4f}—possible leak!")
    if train_auc < 0.7:
        logger.warning(f"⚠️  Overfit check AUC {train_auc:.4f} is low—pipeline may be broken!")

    return checks


def run_experiment(
    csv_path: str,
    model_name: str,
    seeds: list[int],
    output_dir: Path = Path("results"),
) -> list[RunResult]:
    """
    Run the full experiment with multiple seeds.

    For each seed:
      1. Load and deduplicate data
      2. Engineer features (temporal, leak-aware)
      3. Split train/test (80/20 stratified)
      4. Fit preprocessing on train only
      5. Train model
      6. Evaluate on test set
    """
    output_dir.mkdir(exist_ok=True)
    results = []

    # Load data once (reproducible across seeds)
    df, target_rate, n_dedup = load_and_prepare_data(csv_path)

    # Run sanity checks on first seed only (cheap, one-time)
    sanity_results = None

    for i, seed in enumerate(seeds):
        logger.info(f"\n=== Seed {seed} ({i+1}/{len(seeds)}) ===")
        rng = np.random.default_rng(seed)

        # Prepare data (same for all seeds, deterministic from csv)
        df_eng = engineer_features(df)
        X = df_eng.drop('churned', axis=1)
        y = df_eng['churned']

        # Split train/test (seed varies, so split varies)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )

        # Sanity checks on first seed
        if i == 0:
            sanity_results = run_sanity_checks(X_train, y_train, X_test, y_test, seed=seed)

        # Fit preprocessing on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=seed)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, (y_pred_proba > 0.5).astype(int))
        recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))
        f1 = f1_score(y_test, (y_pred_proba > 0.5).astype(int))

        result = RunResult(
            seed=seed,
            model_name=model_name,
            roc_auc_test=roc_auc,
            precision_test=precision,
            recall_test=recall,
            f1_test=f1,
            baseline_auc=sanity_results['baseline_auc'] if sanity_results else 0.0,
            n_train=len(X_train),
            n_test=len(X_test),
            churn_rate=target_rate,
            dedup_rows_removed=n_dedup,
        )
        results.append(result)

        logger.info(f"{model_name}: ROC-AUC={roc_auc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return results


def summarize_results(results_by_model: dict[str, list[RunResult]]) -> dict:
    """
    Compute mean ± std for each model across seeds.
    """
    summary = {}
    for model_name, results in results_by_model.items():
        aucs = [r.roc_auc_test for r in results]
        summary[model_name] = {
            'roc_auc_mean': float(np.mean(aucs)),
            'roc_auc_std': float(np.std(aucs)),
            'roc_auc_values': aucs,
            'n_seeds': len(results),
        }
    return summary
