"""Data loading, cleaning, and splitting for the churn experiment."""
import pandas as pd
import numpy as np

# account_status encodes the target directly ("closed" iff churned == 1)
LEAKY_COLS = ["account_status"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "signup_ordinal"]


def load_and_clean(path: str):
    """
    Load CSV, remove exact duplicates, drop leaky and ID columns.

    Returns (df, info_dict) where info_dict records deduplication stats.
    Exact duplicates must be removed before splitting to prevent the same
    row from appearing in both train and test.
    """
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    # account_status is a perfect label leak: drop before any modeling
    df = df.drop(columns=LEAKY_COLS + ID_COLS)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # Days since reference — used as a numeric feature
    df["signup_ordinal"] = (df[DATE_COL] - pd.Timestamp("2020-01-01")).dt.days

    info = {
        "n_before_dedup": n_before,
        "n_after_dedup": len(df),
        "n_dupes_removed": n_dupes,
    }
    return df, info


def time_split(df: pd.DataFrame, test_fraction: float = 0.30):
    """
    Sort by signup_date and take the earliest rows as train, latest as test.

    A random split would be leaky here: customers with early signup dates
    would appear in test, letting the model implicitly exploit temporal
    structure. The time-based split reflects realistic deployment: the
    model is trained on past customers and evaluated on future ones.
    """
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_fraction))
    train = df_sorted.iloc[:cutoff].copy()
    test = df_sorted.iloc[cutoff:].copy()
    return train, test


def split_xy(df: pd.DataFrame):
    """Return (X as ndarray, y as ndarray)."""
    return df[FEATURES].values, df[TARGET].values
