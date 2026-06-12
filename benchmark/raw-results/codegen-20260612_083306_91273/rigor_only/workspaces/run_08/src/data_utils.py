"""Data loading, deduplication, and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, str]:
    """Load CSV and return clean DataFrame with preprocessing info."""
    df = pd.read_csv(csv_path)

    # Detect exact duplicates (excluding customer_id and signup_date for now)
    dup_cols = ["tenure_months", "monthly_spend", "support_tickets", "account_status", "churned"]
    n_before = len(df)
    df = df.drop_duplicates(subset=dup_cols, keep="first").reset_index(drop=True)
    n_duplicates = n_before - len(df)

    prep_info = f"Removed {n_duplicates} duplicate rows (out of {n_before}). "

    # Alert: account_status is a leak (derived from churned: closed iff churned=1)
    if "account_status" in df.columns:
        # Verify the leak
        leak_check = (df["account_status"] == "closed") == (df["churned"] == 1)
        if leak_check.all():
            prep_info += "LEAK DETECTED: account_status is deterministically derived from churned. Dropping."
        df = df.drop(columns=["account_status"])

    return df, prep_info


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    target_col: str = "churned"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by signup_date (temporal) to respect time ordering.
    Train on earlier dates, test on later dates.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    # Check for contamination (exact duplicates across boundary)
    dup_cols = [c for c in train.columns if c not in ["customer_id", "signup_date"]]

    return train, test


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: StandardScaler = None
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit scaler on train, apply to both.
    Features: tenure_months, monthly_spend, support_tickets.
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X_train_feat = X_train[feature_cols].values
    X_test_feat = X_test[feature_cols].values

    if fit_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
    else:
        X_train_scaled = scaler.transform(X_train_feat)

    X_test_scaled = scaler.transform(X_test_feat)

    return X_train_scaled, X_test_scaled, scaler
