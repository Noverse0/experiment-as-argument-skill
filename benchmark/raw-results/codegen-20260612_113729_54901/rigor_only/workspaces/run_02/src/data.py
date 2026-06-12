import pandas as pd
import numpy as np

TARGET = "churned"
# account_status encodes the target directly ("closed" iff churned==1)
LEAKAGE_COLS = ["account_status"]
ID_COLS = ["customer_id"]
# signup_date is temporal; tenure_months already captures time-in-service
TEMPORAL_COLS = ["signup_date"]
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, remove exact duplicates, drop leakage/ID/temporal columns."""
    df = pd.read_csv(path)

    n_before = len(df)
    # Deduplicate on customer_id BEFORE any train/test split
    df = df.drop_duplicates(subset=["customer_id"])
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[data] Removed {n_removed} duplicate rows ({n_before} -> {len(df)})")

    # Drop direct leakage: account_status is derived from churned
    df = df.drop(columns=LEAKAGE_COLS)
    df = df.drop(columns=ID_COLS)
    df = df.drop(columns=TEMPORAL_COLS)

    return df


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and target vector y as numpy arrays."""
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET].values
    return X, y
