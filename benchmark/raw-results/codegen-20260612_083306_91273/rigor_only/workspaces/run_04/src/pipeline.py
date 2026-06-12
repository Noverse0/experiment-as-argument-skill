"""Data preprocessing pipeline with split-before-transform discipline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_deduplicate(csv_path: str) -> pd.DataFrame:
    """Load CSV and deduplicate exact rows before any analysis."""
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_dropped = n_before - n_after
    return df, n_dropped


def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop account_status (perfect leak: "closed" iff churned=1) and customer_id.
    Keep signup_date for now but don't use it (could engineer days_since_signup if needed).
    """
    return df.drop(columns=['account_status', 'customer_id', 'signup_date'], errors='ignore')


def prepare_features_and_target(df: pd.DataFrame) -> tuple:
    """Extract features and target. No transformations yet (split-before-transform)."""
    y = df['churned'].copy()
    X = df[['tenure_months', 'monthly_spend', 'support_tickets']].copy()
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on train data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaling(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply fitted scaler to any split."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
