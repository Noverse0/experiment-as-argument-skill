"""Data loading and preprocessing with leak awareness."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


FEATURE_NAMES = ["tenure_months", "monthly_spend", "support_tickets"]
LEAK_FEATURES = ["days_since_last_login"]
TARGET_NAME = "churned"


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load CSV and validate structure."""
    df = pd.read_csv(csv_path)
    required_cols = {"customer_id", "signup_date", "tenure_months", "monthly_spend",
                     "support_tickets", "days_since_last_login", "churned"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected {required_cols}, got {set(df.columns)}")
    return df


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Find exact duplicate rows (excluding customer_id which is index-like)."""
    cols = [c for c in df.columns if c != "customer_id"]
    dups = df[df.duplicated(subset=cols, keep=False)].sort_values(cols)
    return dups


def deduplicate_before_split(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicates, keeping first occurrence. Returns (deduplicated_df, num_removed)."""
    cols = [c for c in df.columns if c != "customer_id"]
    initial_len = len(df)
    df_dedup = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    removed = initial_len - len(df_dedup)
    return df_dedup, removed


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Stratified train/test split on target."""
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame,
               use_scaling: bool = True) -> tuple:
    """Fit scaler on train, apply to test. Returns (X_train_transformed, X_test_transformed, scaler)."""
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train.values, X_test.values, None


def get_class_distribution(y: pd.Series) -> dict:
    """Return target class distribution."""
    counts = y.value_counts().to_dict()
    total = len(y)
    return {
        "class_0_count": counts.get(0, 0),
        "class_1_count": counts.get(1, 0),
        "class_1_rate": counts.get(1, 0) / total if total > 0 else 0.0,
        "total": total,
    }
