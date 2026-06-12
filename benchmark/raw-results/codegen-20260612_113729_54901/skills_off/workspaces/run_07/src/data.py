"""Data loading, deduplication, feature engineering, and splitting."""
import pandas as pd


def load_data(path: str) -> tuple[pd.DataFrame, int]:
    """Load CSV, remove duplicates, drop leaky columns, engineer features.

    Drops account_status (derived from target — perfect label leak).
    Converts signup_date to days_since_start and sorts by it for temporal splits.
    Returns (cleaned_df, n_duplicates_removed).
    """
    df = pd.read_csv(path)

    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    df = df.drop(columns=["customer_id"])

    # account_status encodes the target: "closed" iff churned==1 — drop it.
    df = df.drop(columns=["account_status"])

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    min_date = df["signup_date"].min()
    df["days_since_start"] = (df["signup_date"] - min_date).dt.days
    df = df.drop(columns=["signup_date"])

    df = df.sort_values("days_since_start").reset_index(drop=True)

    return df, n_dupes


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split: earliest train_frac rows → train, rest → test.

    Requires df already sorted by days_since_start (load_data guarantees this).
    """
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
