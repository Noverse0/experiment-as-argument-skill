"""Data loading, deduplication, and splitting for the churn experiment."""
import pandas as pd

# days_since_last_login is a TARGET LEAK: it is derived from the churned outcome
# (churned customers stop logging in, so this value is recorded after the outcome
# is determined). Including it would give artificially inflated metrics.
# signup_date is temporal: used only to order the time-based split, not as a feature.
# customer_id is an ID column with no predictive signal.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"

DROPPED = {
    "customer_id": "ID column — no signal",
    "signup_date": "temporal — used for split ordering only",
    "days_since_last_login": "target leak — derived from churned outcome",
}


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Must be called BEFORE any split."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        print(f"Dedup: removed {n_removed} exact duplicate rows ({n_before} → {len(df)})")
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Sort by signup_date and use the last test_frac as the test set.

    Using a temporal split avoids the situation where duplicate rows from the
    same calendar period straddle train/test, and mirrors production deployment
    (model trained on earlier cohorts, evaluated on newer ones).
    """
    df_sorted = df.sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:cutoff].reset_index(drop=True)
    test = df_sorted.iloc[cutoff:].reset_index(drop=True)
    return train, test


def get_X_y(df: pd.DataFrame):
    return df[FEATURES].copy(), df[TARGET].copy()
