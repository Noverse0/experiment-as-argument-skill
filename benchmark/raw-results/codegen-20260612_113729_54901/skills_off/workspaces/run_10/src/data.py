"""Data loading, cleaning, and splitting for the churn experiment."""
import pandas as pd

# account_status encodes the target directly ("closed" iff churned) — perfect leak.
# customer_id is a row identifier with no predictive value.
# signup_date is used only to impose temporal ordering for the split.
LEAKY_COLS = ["account_status", "customer_id", "signup_date"]
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"
TEMPORAL_COL = "signup_date"


def load_and_clean(csv_path: str) -> tuple:
    """Load CSV, remove exact duplicates, return (df, metadata_dict).

    Deduplication must happen before any split to prevent identical rows
    from appearing in both train and test sets.
    """
    df = pd.read_csv(csv_path)
    df[TEMPORAL_COL] = pd.to_datetime(df[TEMPORAL_COL])

    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_deduped = len(df)

    meta = {
        "n_raw": n_raw,
        "n_deduped": n_deduped,
        "n_duplicates_dropped": n_raw - n_deduped,
        "churn_rate": float(df[TARGET_COL].mean()),
    }
    return df, meta


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Sort by signup_date; earliest rows → train, latest rows → test.

    Random splitting on temporal data would allow a model trained on
    'future' customers to predict 'past' customers — a time-leakage form.
    The temporal column is excluded from feature matrices.
    """
    df = df.copy()
    df[TEMPORAL_COL] = pd.to_datetime(df[TEMPORAL_COL])
    df = df.sort_values(TEMPORAL_COL).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_frac))
    cutoff_date = str(df[TEMPORAL_COL].iloc[split_idx])

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].reset_index(drop=True)
    X_test = test_df[FEATURE_COLS].reset_index(drop=True)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_test = test_df[TARGET_COL].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, cutoff_date
