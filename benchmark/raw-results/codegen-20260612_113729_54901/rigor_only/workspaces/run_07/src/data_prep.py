"""Data loading, deduplication, and train/test splitting."""

import pandas as pd


LEAKY_COLS = ["account_status"]  # derived from target: "closed" iff churned==1
ID_COLS = ["customer_id"]
TEMPORAL_COL = "signup_date"
TARGET = "churned"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TEMPORAL_COL])

    # Drop exact duplicates before any split so they can't straddle train/test.
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    print(f"[data_prep] dropped {n_dupes} exact duplicate rows ({n_before} → {len(df)})")

    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.80):
    """Chronological split: train on earlier signups, test on later ones.

    Random splits on temporal data are leakage — future customers could appear
    in the training set, which is impossible in production.
    """
    df_sorted = df.sort_values(TEMPORAL_COL).reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:cutoff_idx].copy()
    test = df_sorted.iloc[cutoff_idx:].copy()
    print(
        f"[data_prep] time split: train={len(train)} "
        f"(up to {train[TEMPORAL_COL].max().date()}), "
        f"test={len(test)} "
        f"(from {test[TEMPORAL_COL].min().date()})"
    )
    return train, test


def get_X_y(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def churn_rate(df: pd.DataFrame) -> float:
    return df[TARGET].mean()
