"""Data loading and cleaning for the churn experiment."""
import pandas as pd
import numpy as np

# days_since_last_login is a target leak: it's recorded after the outcome
# (churned customers stop logging in, so the value is derived from churn itself).
LEAK_FEATURES = ["days_since_last_login"]

# customer_id is an opaque identifier; signup_date drives the temporal split
# but is not a predictive feature for the model.
DROP_FEATURES = ["customer_id", "signup_date"] + LEAK_FEATURES

TARGET = "churned"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_date"])

    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    print(f"Deduplication: removed {n_removed} rows ({n_before} → {len(df)})")

    # Sort ascending by signup_date so TimeSeriesSplit respects temporal order.
    df = df.sort_values("signup_date").reset_index(drop=True)
    return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
