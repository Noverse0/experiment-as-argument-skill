"""Load and preprocess the churn dataset with proper data discipline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_deduplicate(csv_path: str) -> pd.DataFrame:
    """Load dataset and remove exact duplicates.

    The dataset intentionally includes 200 exact duplicate rows.
    Removing them before split prevents leakage across train/test boundary.
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    n_after = len(df)
    n_removed = n_before - n_after
    print(f"Loaded {n_before} rows, removed {n_removed} exact duplicates, {n_after} remain.")
    return df


def prepare_features(df: pd.DataFrame, fit_scaler: bool = False, scaler: StandardScaler = None):
    """
    Extract features from raw data. Split before transform: scaler is fit on train only.

    Feature selection:
    - tenure_months: legitimate signal
    - monthly_spend: legitimate signal
    - support_tickets: legitimate signal
    - signup_month, signup_year, days_since_signup: temporal features from signup_date

    Excluded:
    - customer_id: identifier, not a feature
    - days_since_last_login: TARGET LEAKAGE (derived from churned status)
    - signup_date: converted to features, then dropped
    """
    df = df.copy()

    # Parse signup_date and extract temporal features
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    reference_date = pd.Timestamp("2024-01-01")  # Fixed reference for consistency
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_year"] = df["signup_date"].dt.year
    df["days_since_signup"] = (reference_date - df["signup_date"]).dt.days

    # Select features (exclude customer_id, signup_date, days_since_last_login, churned)
    feature_cols = [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "signup_month",
        "signup_year",
        "days_since_signup",
    ]
    X = df[feature_cols].copy()
    y = df["churned"].values  # Convert to numpy array

    # Scale features. If fit_scaler=True, fit new scaler. Otherwise, apply provided scaler.
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values  # numpy array

    return X, y, feature_cols, scaler


def split_and_prepare(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load, deduplicate, split (stratified), and prepare features.

    Returns:
        X_train, X_test, y_train, y_test, feature_cols, scaler
    """
    df = load_and_deduplicate(csv_path)

    # Stratified split to respect class imbalance
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df["churned"], random_state=random_state
    )

    # Fit scaler on train, apply to both
    X_train, y_train, feature_cols, scaler = prepare_features(df_train, fit_scaler=True)
    X_test, y_test, _, _ = prepare_features(df_test, fit_scaler=False, scaler=scaler)

    print(f"Split: {len(df_train)} train, {len(df_test)} test")
    print(f"Class balance - train: {y_train.mean():.3f} churned, test: {y_test.mean():.3f} churned")

    return X_train, X_test, y_train, y_test, feature_cols, scaler
