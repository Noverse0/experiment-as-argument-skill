import pandas as pd
import numpy as np

LEAKY_COLS = ["account_status"]
ID_COLS = ["customer_id"]
TARGET_COL = "churned"
# Fixed reference date — no fitting needed, just a shift
DATE_REF = pd.Timestamp("2023-01-01")


def load_and_clean(path: str):
    """Load CSV, remove exact duplicates, sort by signup_date.

    Returns (df, stats) where stats reports rows removed by deduplication.
    Deduplication happens BEFORE any split to prevent train/test contamination
    from duplicate rows straddling the boundary.
    """
    df = pd.read_csv(path, parse_dates=["signup_date"])
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_clean = len(df)
    df = df.sort_values("signup_date").reset_index(drop=True)
    stats = {
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_duplicates_removed": n_raw - n_clean,
    }
    return df, stats


def make_features(df: pd.DataFrame):
    """Drop leaky and ID columns; encode signup_date numerically.

    account_status is a perfect leak (value is 'closed' iff churned==1).
    customer_id is an identifier with no predictive signal.
    signup_date is encoded as days since a fixed reference so no fitting is needed.
    """
    drop_cols = LEAKY_COLS + ID_COLS + [TARGET_COL]
    feat = df.drop(columns=drop_cols).copy()
    feat["days_since_ref"] = (feat["signup_date"] - DATE_REF).dt.days
    feat = feat.drop(columns=["signup_date"])
    X = feat.values.astype(float)
    y = df[TARGET_COL].values
    return X, y, feat.columns.tolist()
