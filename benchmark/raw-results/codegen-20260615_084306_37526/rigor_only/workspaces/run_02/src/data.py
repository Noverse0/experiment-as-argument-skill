"""Data loading, cleaning, and feature/target definitions."""
import pandas as pd
import numpy as np

# days_since_last_login is recorded AFTER the outcome (churned customers stop
# logging in by definition), making it a target leak despite the plausible name.
LEAK_COLS = ["days_since_last_login"]
ID_COLS = ["customer_id", "signup_date"]
TARGET = "churned"
# Only the causally legitimate features remain after removing leakage and IDs.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate and record cleaning stats."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)
    stats = {
        "n_before_dedup": n_before,
        "n_after_dedup": len(df),
        "n_duplicates_removed": n_dupes,
    }
    return df, stats


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def churn_rate(y: pd.Series) -> float:
    return float(y.mean())
