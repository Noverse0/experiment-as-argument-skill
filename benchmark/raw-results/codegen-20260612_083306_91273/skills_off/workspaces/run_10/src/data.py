"""Data loading, deduplication, and preprocessing."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV, drop leaky features, deduplicate, extract temporal features."""
    df = pd.read_csv(csv_path)

    # Drop account_status: perfectly leaked from the target (churned).
    # In the data generator: account_status = "closed" iff churned == 1
    df = df.drop(columns=["account_status"])

    # Drop customer_id: just an identifier, not predictive.
    df = df.drop(columns=["customer_id"])

    # Extract temporal feature from signup_date before dropping it.
    # Days since signup at the point of data collection.
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    max_date = df["signup_date"].max()
    df["days_since_signup"] = (max_date - df["signup_date"]).dt.days
    df = df.drop(columns=["signup_date"])

    # Report and remove exact duplicates.
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_dups = n_before - n_after
    print(f"[DATA] Removed {n_dups} exact duplicates ({n_before} -> {n_after} rows)")

    return df


def stratified_split(df: pd.DataFrame, train_size: float = 0.6, val_size: float = 0.2,
                     random_state: int = 42) -> tuple:
    """
    Split into train/val/test with stratification on target.

    Args:
        df: DataFrame with 'churned' column as target
        train_size: fraction for training (0.6 -> 60%)
        val_size: fraction for validation (0.2 -> 20%, rest goes to test)
        random_state: seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=["churned"])
    y = df["churned"]

    # First split: train vs (val + test)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )

    # Second split: val vs test from the rest
    # val_size is relative to original, so relative to rest it becomes val_size / (1 - train_size)
    val_fraction = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=val_fraction, stratify=y_rest, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Fit preprocessing on train only, apply to val/test.

    Handles:
    - Scaling numerical features
    - One-hot encoding categorical features (if any)

    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled)
    """
    # Identify feature types
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Fit scaler on train
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    if numerical_cols:
        scaler.fit(X_train[numerical_cols])
        X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
        X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # One-hot encode categorical (fit on train only)
    if categorical_cols:
        X_train_scaled = pd.get_dummies(X_train_scaled, columns=categorical_cols, drop_first=True)
        X_val_scaled = pd.get_dummies(X_val_scaled, columns=categorical_cols, drop_first=True)
        X_test_scaled = pd.get_dummies(X_test_scaled, columns=categorical_cols, drop_first=True)

        # Ensure all three have the same columns
        train_cols = set(X_train_scaled.columns)
        val_cols = set(X_val_scaled.columns)
        test_cols = set(X_test_scaled.columns)
        all_cols = train_cols | val_cols | test_cols

        for col in all_cols:
            if col not in X_train_scaled.columns:
                X_train_scaled[col] = 0
            if col not in X_val_scaled.columns:
                X_val_scaled[col] = 0
            if col not in X_test_scaled.columns:
                X_test_scaled[col] = 0

        # Align column order
        X_train_scaled = X_train_scaled[sorted(all_cols)]
        X_val_scaled = X_val_scaled[sorted(all_cols)]
        X_test_scaled = X_test_scaled[sorted(all_cols)]

    return X_train_scaled, X_val_scaled, X_test_scaled
