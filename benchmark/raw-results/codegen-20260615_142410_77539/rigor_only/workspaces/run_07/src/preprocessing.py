"""Data preprocessing and validation for churn experiment."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load dataset and perform basic validation."""
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = {
        "customer_id", "signup_date", "tenure_months", "monthly_spend",
        "support_tickets", "days_since_last_login", "churned"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check target distribution
    target_rate = df["churned"].mean()
    print(f"Target rate (churn): {target_rate:.3f} ({df['churned'].sum()}/{len(df)})")

    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify exact duplicates, excluding customer_id."""
    dup_cols = [c for c in df.columns if c != "customer_id"]
    duplicates = df[df.duplicated(subset=dup_cols, keep=False)]
    print(f"Found {len(duplicates)} rows in {len(duplicates) // 2} duplicate pairs")
    return duplicates


def preprocess_features(df: pd.DataFrame, fit_scaler=None, feature_set="clean") -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess features for modeling.

    feature_set:
      - "clean": drop days_since_last_login (target leakage)
      - "leaked": include days_since_last_login (for leakage detection)
    """
    df_work = df.copy()

    # Extract day-count from signup_date
    df_work["signup_date"] = pd.to_datetime(df_work["signup_date"])
    reference_date = df_work["signup_date"].max()
    df_work["days_since_signup"] = (reference_date - df_work["signup_date"]).dt.days

    # Select features based on feature_set
    if feature_set == "clean":
        feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "days_since_signup"]
    elif feature_set == "leaked":
        feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "days_since_signup", "days_since_last_login"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X = df_work[feature_cols].values
    y = df_work["churned"].values

    # Fit or apply scaler
    if fit_scaler:
        scaler = fit_scaler
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def create_stratified_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = None) -> Tuple:
    """Create stratified train/test split preserving target distribution."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
