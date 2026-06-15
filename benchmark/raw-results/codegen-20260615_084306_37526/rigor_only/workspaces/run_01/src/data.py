"""Load, clean, and prepare the churn dataset for modeling."""
import pandas as pd

# Only the three causally honest features + seasonality proxy.
# days_since_last_login is deliberately excluded: it is a post-outcome leak
# (churned customers have, by definition, stopped logging in, so this value is
# recorded after the outcome is determined). Including it would inflate AUC
# without reflecting real predictive power at decision time.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "signup_month"]
TARGET = "churned"


def load_and_prepare(path: str) -> tuple:
    """Return (X, y, info) with duplicates removed, leaks dropped, sorted by time.

    Data is sorted by signup_date so that TimeSeriesSplit respects temporal order.
    """
    df = pd.read_csv(path, parse_dates=["signup_date"])

    # Remove the 200 injected exact-duplicate rows before any split.
    # Duplicates share all column values; straddling train/test would constitute
    # indirect target leakage through memorised rows.
    n_before = len(df)
    df = df.drop_duplicates().copy()
    n_dropped = n_before - len(df)

    # Drop post-outcome leak.
    df = df.drop(columns=["days_since_last_login"])

    # Drop bare identifier.
    df = df.drop(columns=["customer_id"])

    # Sort ascending by signup_date so temporal CV folds are valid.
    df = df.sort_values("signup_date").reset_index(drop=True)

    # Extract seasonality proxy; drop the raw date column.
    df["signup_month"] = df["signup_date"].dt.month
    df = df.drop(columns=["signup_date"])

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y, {"n_dropped_duplicates": n_dropped}
