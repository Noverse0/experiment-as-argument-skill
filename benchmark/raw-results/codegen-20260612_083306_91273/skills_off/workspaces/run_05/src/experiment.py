"""Core experiment for churn prediction: gradient boosting vs logistic regression."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RunResult:
    """Result from a single seed run."""
    seed: int
    model_name: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


@dataclass
class SanityCheckResult:
    """Result from sanity checks."""
    baseline_accuracy: float
    label_shuffle_accuracy: float
    overfit_train_loss: float
    has_duplicates: int


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and drop leaked/unwanted columns. Return clean dataframe."""
    df = pd.read_csv(csv_path)

    # Drop account_status (perfectly derived from target - leak!)
    # Drop customer_id (identifier, not a feature)
    # Drop signup_date for now (temporal, but will derive features from it)
    original_cols = set(df.columns)
    df['days_since_signup'] = (pd.to_datetime(df['signup_date']) - pd.Timestamp("2023-01-01")).dt.days
    df = df.drop(columns=['account_status', 'customer_id', 'signup_date'])

    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows."""
    return df.duplicated().sum()


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Splits before any fitting operations to prevent leakage.
    Respects stratification to maintain class balance.
    """
    y = df['churned']
    X = df.drop(columns=['churned'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler on train set only, apply to both.

    All fit-like operations (scaling, encoding) happen after split,
    fitted on train only, then applied to test.
    """
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit scaler ONLY on train
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    # Apply to test with train's statistics
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled.values, X_test_scaled.values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics. Use AUC-ROC as primary metric (robust to class imbalance)."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
    }


def run_single_seed(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    model_class, seed: int
) -> RunResult:
    """Train one model on this seed. Return metrics."""
    model = model_class(random_state=seed)

    # Set additional seeds for ensemble models
    if hasattr(model, 'init'):
        model.init = 'zero'

    model.fit(X_train, y_train)

    # Train metrics
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

    # Test metrics
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    model_name = model.__class__.__name__
    return RunResult(
        seed=seed,
        model_name=model_name,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )


def run_experiment(csv_path: str, num_seeds: int = 5) -> Tuple[list, dict]:
    """
    Run full experiment with multiple seeds.

    Returns: (list of RunResult, sanity check results)
    """
    df = load_and_preprocess_data(csv_path)

    # Sanity check: duplicates
    n_dups = check_duplicates(df)

    # Baseline: majority class accuracy
    baseline_acc = max(df['churned'].value_counts()) / len(df)

    # Label shuffle sanity check
    y_shuffled = df['churned'].copy().sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_shuffled = df.copy()
    df_shuffled['churned'] = y_shuffled.values
    X_train_s, X_test_s, y_train_s, y_test_s = split_data(df_shuffled, seed=42)
    X_train_s, X_test_s = preprocess_features(X_train_s, X_test_s)

    model_s = LogisticRegression(random_state=42, max_iter=1000)
    model_s.fit(X_train_s, y_train_s)
    y_test_pred_s = model_s.predict(X_test_s)
    label_shuffle_acc = accuracy_score(y_test_s, y_test_pred_s)

    # Overfit check: can we reach ~0 loss on tiny subset?
    df_tiny = df.sample(n=min(100, len(df)), random_state=42)
    X_tiny = df_tiny.drop(columns=['churned']).values
    y_tiny = df_tiny['churned'].values
    scaler_tiny = StandardScaler()
    X_tiny = scaler_tiny.fit_transform(X_tiny)
    model_tiny = LogisticRegression(random_state=42, max_iter=1000)
    model_tiny.fit(X_tiny, y_tiny)
    y_pred_tiny = model_tiny.predict(X_tiny)
    overfit_acc = accuracy_score(y_tiny, y_pred_tiny)

    sanity = SanityCheckResult(
        baseline_accuracy=baseline_acc,
        label_shuffle_accuracy=label_shuffle_acc,
        overfit_train_loss=overfit_acc,
        has_duplicates=n_dups,
    )

    # Run experiment with multiple seeds
    results = []
    seeds = range(42, 42 + num_seeds)

    for seed in seeds:
        X_train, X_test, y_train, y_test = split_data(df, seed=seed)
        X_train, X_test = preprocess_features(X_train, X_test)

        # LogisticRegression
        lr_result = run_single_seed(X_train, X_test, y_train, y_test, LogisticRegression, seed)
        results.append(lr_result)

        # GradientBoostingClassifier
        gb_result = run_single_seed(X_train, X_test, y_train, y_test, GradientBoostingClassifier, seed)
        results.append(gb_result)

    return results, sanity


def summarize_results(results: list, sanity: dict) -> Dict:
    """Aggregate results across seeds and format for report."""
    by_model = {}

    for result in results:
        model = result.model_name
        if model not in by_model:
            by_model[model] = {'test_auc_roc': [], 'test_f1': [], 'test_accuracy': []}

        by_model[model]['test_auc_roc'].append(result.test_metrics['auc_roc'])
        by_model[model]['test_f1'].append(result.test_metrics['f1'])
        by_model[model]['test_accuracy'].append(result.test_metrics['accuracy'])

    summary = {}
    for model, metrics in by_model.items():
        summary[model] = {
            'test_auc_roc_mean': float(np.mean(metrics['test_auc_roc'])),
            'test_auc_roc_std': float(np.std(metrics['test_auc_roc'])),
            'test_f1_mean': float(np.mean(metrics['test_f1'])),
            'test_f1_std': float(np.std(metrics['test_f1'])),
            'test_accuracy_mean': float(np.mean(metrics['test_accuracy'])),
            'test_accuracy_std': float(np.std(metrics['test_accuracy'])),
            'n_seeds': len(metrics['test_auc_roc']),
        }

    return {
        'models': summary,
        'sanity_checks': {
            'baseline_accuracy': float(sanity.baseline_accuracy),
            'label_shuffle_accuracy': float(sanity.label_shuffle_accuracy),
            'overfit_train_accuracy': float(sanity.overfit_train_loss),
            'duplicate_rows': int(sanity.has_duplicates),
        }
    }
