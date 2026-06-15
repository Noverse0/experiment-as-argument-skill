"""Data loading, cleaning, and preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and remove duplicates.

    Removes 200 exact duplicates planted in the dataset.
    """
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    df = df.drop_duplicates(ignore_index=True)
    removed = initial_rows - len(df)
    print(f"Loaded {initial_rows} rows, removed {removed} duplicates, {len(df)} remain")
    return df


def audit_leak_surface(df: pd.DataFrame) -> None:
    """Log observations about leak surface (for post-run analysis only)."""
    print("\nLeak surface audit (post-run, for reference):")
    print(f"  days_since_last_login mean (churned=0): {df[df['churned']==0]['days_since_last_login'].mean():.1f}")
    print(f"  days_since_last_login mean (churned=1): {df[df['churned']==1]['days_since_last_login'].mean():.1f}")
    print(f"  Class balance: {df['churned'].mean():.1%} churn rate")


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features, drop customer_id and the leak.

    Returns:
        (X, y, feature_names) where X is the feature matrix and y is the target.

    Drops:
    - customer_id: just an identifier
    - days_since_last_login: target leak (derivable from churned status)
    - signup_date: temporal column, not used (time-aware evaluation is via CV order, not feature)
    """
    X = df[["tenure_months", "monthly_spend", "support_tickets"]].copy()
    y = df["churned"].copy()
    feature_names = list(X.columns)

    print(f"\nFeatures used: {feature_names}")
    print(f"Features dropped: customer_id (identifier), days_since_last_login (target leak), signup_date (temporal, not featured)")

    return X, y, feature_names


def get_cv_splitter(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Return a stratified K-fold splitter.

    Stratification preserves class balance in each fold.
    Random state ensures reproducibility.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def preprocess_for_lr(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale features for logistic regression.

    Fits scaler on train only, applies to test. No leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def preprocess_for_gb(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """No preprocessing for gradient boosting (it handles feature scaling naturally)."""
    return X_train.values, X_test.values
