"""Data loading, cleaning, and feature engineering for the churn experiment."""

import pandas as pd

# days_since_last_login is recorded after the outcome: a churned customer has
# already stopped logging in, so this value is derived from the label.
# Including it would inflate all metrics and invalidate the comparison.
LEAK_COLS = ["days_since_last_login"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"
_REFERENCE_DATE = pd.Timestamp("2023-01-01")


def load_and_clean(path: str) -> tuple:
    """Load CSV and remove exact duplicate rows.

    Returns (cleaned_df, n_duplicates_removed).
    Duplicates must be removed before any split to prevent rows from
    straddling the train/test boundary.
    """
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates()
    return df.reset_index(drop=True), n_before - len(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leak/ID columns and convert signup_date to a numeric feature."""
    df = df.copy()
    df["signup_days"] = (pd.to_datetime(df[DATE_COL]) - _REFERENCE_DATE).dt.days
    drop_cols = ID_COLS + LEAK_COLS + [DATE_COL]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def make_pipeline(estimator):
    """Wrap estimator in a StandardScaler pipeline (scaler fitted on train fold only)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
