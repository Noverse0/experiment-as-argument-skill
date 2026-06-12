import pandas as pd
import numpy as np


def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """Load CSV, remove leakage sources, and deduplicate.

    Traps handled:
    - account_status: "closed" iff churned=1 — perfect target leak, dropped.
    - customer_id: identifier with no predictive value, dropped.
    - 200 planted duplicate rows: removed before splitting to prevent
      the same row appearing in both train and test.
    """
    df = pd.read_csv(path)
    audit: dict = {}

    df = df.drop(columns=["customer_id"])

    # account_status encodes the target perfectly ("closed" iff churned=1).
    # Retaining it would cause near-100% AUC with zero real predictive signal.
    df = df.drop(columns=["account_status"])
    audit["dropped_cols"] = ["customer_id", "account_status"]

    n_before = len(df)
    df = df.drop_duplicates()
    audit["duplicates_removed"] = n_before - len(df)

    # Convert signup_date to numeric days since dataset start.
    # Kept as a feature: captures cohort effects and seasonal patterns.
    df = df.copy()
    df["signup_days"] = (
        pd.to_datetime(df["signup_date"]) - pd.Timestamp("2023-01-01")
    ).dt.days
    df = df.drop(columns=["signup_date"])

    return df, audit


def temporal_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by signup_days and cut at a fixed percentile.

    signup_date is temporal. A random split would let later-cohort customers
    appear in train while earlier ones appear in test — time leakage.
    Using an ordered cut ensures train = earlier cohorts, test = later cohorts.
    """
    df_sorted = df.sort_values("signup_days").reset_index(drop=True)
    n_train = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:n_train].copy(), df_sorted.iloc[n_train:].copy()
