"""Data pipeline: loading, splitting, preprocessing, and evaluation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset and remove exact duplicates."""
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicate rows. {len(df)} rows remaining.")
    return df


def time_split(
    df: pd.DataFrame, train_fraction: float = 0.7, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by signup_date to respect temporal order.
    No information leakage from future into past.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def get_clean_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract clean features (no leakage) and target.
    Excludes: customer_id, signup_date, days_since_last_login (LEAK).
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = df[feature_cols].values.astype(float)
    y = df["churned"].values.astype(int)
    return X, y


def get_features_with_leak(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features including the leak (days_since_last_login).
    Used only for leakage ceiling check.
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "days_since_last_login"]
    X = df[feature_cols].values.astype(float)
    y = df["churned"].values.astype(int)
    return X, y


def preprocess(
    X_train: np.ndarray, X_test: np.ndarray, fit_only: bool = False
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features. Fit scaler on train, apply to test.
    If fit_only=True, only scale train (for sanity checks).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if not fit_only else None
    return X_train_scaled, X_test_scaled, scaler


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> dict:
    """Compute metrics: AUC, precision, recall, F1, log-loss."""
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, y_pred_proba),
    }


def get_baseline_metrics(y_test: np.ndarray) -> dict:
    """Baseline: always predict majority class."""
    majority_class = np.argmax(np.bincount(y_test))
    y_pred = np.full_like(y_test, majority_class)
    # Baseline proba for class 1: high if majority is 1, low if majority is 0
    y_pred_proba = np.full(len(y_test), 0.99 if majority_class == 1 else 0.01, dtype=float)
    return evaluate(y_test, y_pred, y_pred_proba), majority_class


def check_target_distribution(df: pd.DataFrame) -> None:
    """Verify target distribution."""
    churn_rate = df["churned"].mean()
    print(f"Target distribution: {churn_rate:.2%} churn, {1-churn_rate:.2%} no-churn")
    print(f"  Positive examples: {(df['churned'] == 1).sum()}")
    print(f"  Negative examples: {(df['churned'] == 0).sum()}")
