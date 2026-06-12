"""Data loading, cleaning, and feature extraction.

Leakage mitigations applied here (documented explicitly):
- account_status is dropped: it encodes the label (closed ↔ churned=1), making it a
  perfect target leak. Any model trained with it would exploit the label directly.
- Exact duplicates are removed before splitting: the dataset contains 200 appended
  duplicate rows. With a random split these straddle train/test, letting the model
  memorise and "predict" held-out rows it already saw during training.
- customer_id is dropped: identifier with no predictive content.
- signup_date is retained only for temporal ordering of splits; it is not passed as a
  model feature because its generative role is absorbed by tenure_months.
"""

import pandas as pd

LEAK_COLS = ["account_status"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[DATE_COL])


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}

    # Drop label-leaking column
    df = df.drop(columns=LEAK_COLS + ID_COLS, errors="ignore")

    # Remove exact duplicates (must happen before any split)
    before = len(df)
    df = df.drop_duplicates()
    stats["n_duplicates_removed"] = before - len(df)

    # Sort by signup_date so temporal splits are well-defined
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    stats["n_rows"] = len(df)
    stats["churn_rate"] = float(df[TARGET].mean())
    return df, stats


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()
