"""Load, validate, and split the churn dataset with leak detection."""
import pandas as pd
import numpy as np
from typing import Tuple


def load_data(path: str = "churn.csv") -> pd.DataFrame:
    """Load CSV and validate structure."""
    df = pd.read_csv(path)
    assert "churned" in df.columns, "Missing target column 'churned'"
    assert len(df) > 0, "Empty dataset"
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows."""
    dup_count = df.duplicated().sum()
    return dup_count


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping first."""
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicates")
    return df


def detect_leaks(df: pd.DataFrame) -> list:
    """Identify suspected leaked features by examining their relationship to target."""
    target = df["churned"]
    leaks = []

    # Check days_since_last_login: should be strongly predictive if leaked
    if "days_since_last_login" in df.columns:
        feature = df["days_since_last_login"]
        churn_mean = feature[target == 1].mean()
        non_churn_mean = feature[target == 0].mean()
        ratio = churn_mean / non_churn_mean if non_churn_mean > 0 else 0
        # If churned customers have 5x more days, it's likely derived from outcome
        if ratio > 3.0:
            leaks.append(
                f"days_since_last_login: churned={churn_mean:.1f}d, "
                f"non-churn={non_churn_mean:.1f}d (ratio={ratio:.1f}). "
                "LIKELY LEAK: derived from outcome."
            )

    return leaks


def prepare_features(df: pd.DataFrame, drop_leaks: bool = True) -> Tuple[pd.DataFrame, list]:
    """Select and prepare features. Returns (feature_df, feature_names)."""
    # Honest features with causal signal
    clean_features = ["tenure_months", "monthly_spend", "support_tickets"]

    # Leaked feature (strong but derived from outcome)
    leak_features = ["days_since_last_login"]

    # Temporal feature (safe to use for ordering/splitting, not as direct input)
    temporal_features = ["signup_date"]

    if drop_leaks:
        selected = clean_features
    else:
        selected = clean_features + leak_features

    X = df[selected].copy()
    return X, selected


def time_based_split(
    df: pd.DataFrame,
    temporal_col: str = "signup_date",
    train_ratio: float = 0.8,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time. Ensures temporal consistency.

    Args:
        df: dataframe with temporal column
        temporal_col: column to sort by
        train_ratio: fraction for training
        random_state: unused (for API consistency)

    Returns:
        (train_df, test_df)
    """
    df_sorted = df.sort_values(temporal_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train, test


def get_split(
    path: str = "churn.csv",
    train_ratio: float = 0.8,
    random_state: int = 42,
    drop_leaks: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Full pipeline: load → dedup → leak detection → prepare → split.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    df = load_data(path)

    # Check duplicates and remove
    dup_count = check_duplicates(df)
    print(f"Found {dup_count} duplicate rows")
    df = remove_duplicates(df)

    # Detect leaks
    leaks = detect_leaks(df)
    if leaks:
        for leak in leaks:
            print(f"⚠️  LEAK DETECTED: {leak}")

    # Time-based split before preparing features
    train, test = time_based_split(df, temporal_col="signup_date", train_ratio=train_ratio)

    # Prepare features
    X_train, feature_names = prepare_features(train, drop_leaks=drop_leaks)
    X_test, _ = prepare_features(test, drop_leaks=drop_leaks)
    y_train = train["churned"].values
    y_test = test["churned"].values

    # Validate split
    assert len(X_train) > 0, "Train set is empty"
    assert len(X_test) > 0, "Test set is empty"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"

    print(f"Split: {len(train)} train, {len(test)} test")
    print(f"Features: {feature_names}")
    print(f"Churn rate: train={y_train.mean():.1%}, test={y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, feature_names
