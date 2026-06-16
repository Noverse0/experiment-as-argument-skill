"""Data loading and feature engineering for churn prediction."""
import pandas as pd
import numpy as np


HONEST_FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
LEAKY_FEATURES = ["days_since_last_login"]
TARGET = "churned"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw churn CSV."""
    df = pd.read_csv(csv_path)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target, dropping leaky features.

    Features: tenure_months, monthly_spend, support_tickets.
    Excludes days_since_last_login (timing leak).
    """
    X = df[HONEST_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def time_based_split(
    df: pd.DataFrame, test_month: int = 10, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by signup_date month: before test_month is train, after is test.

    Args:
        df: DataFrame with signup_date column (YYYY-MM-DD format).
        test_month: Months >= this go to test (1-12).
        seed: For reproducibility (not used in time split but for consistency).

    Returns:
        (train_df, test_df) split by month.
    """
    df = df.copy()
    df["signup_month"] = pd.to_datetime(df["signup_date"]).dt.month

    train = df[df["signup_month"] < test_month].drop("signup_month", axis=1)
    test = df[df["signup_month"] >= test_month].drop("signup_month", axis=1)

    return train, test


def get_class_distribution(y: pd.Series) -> dict:
    """Return churn rate and counts."""
    counts = y.value_counts()
    total = len(y)
    return {
        "n_samples": total,
        "n_churn": int(counts.get(1, 0)),
        "n_no_churn": int(counts.get(0, 0)),
        "churn_rate": float(counts.get(1, 0) / total) if total > 0 else 0.0,
    }
