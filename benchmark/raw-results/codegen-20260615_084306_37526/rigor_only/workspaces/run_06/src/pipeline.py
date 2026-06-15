"""Data loading, feature engineering, and model construction."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Post-outcome feature: churned customers have stopped logging in, so
# days_since_last_login is recorded after the churn outcome — including it
# would leak the label into the features.
LEAKY_COLS = ["days_since_last_login"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"


def load_and_clean(path: str) -> tuple:
    """Load CSV, remove exact duplicate rows, return (df, n_dupes_removed)."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def build_features(df: pd.DataFrame) -> tuple:
    """Return (X, y, signup_dates).

    Drops leaky columns and identifiers. Converts signup_date to an ordinal
    numeric feature (days from the earliest signup in the dataset) so that
    cohort effects can be captured without carrying a raw date string.
    """
    dates = pd.to_datetime(df[DATE_COL])
    ref = dates.min()
    df = df.copy()
    df["signup_days"] = (dates - ref).dt.days

    drop_cols = ID_COLS + LEAKY_COLS + [DATE_COL, TARGET]
    X = df.drop(columns=drop_cols)
    y = df[TARGET]
    return X, y, dates


def temporal_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                   train_frac: float = 0.80) -> tuple:
    """Chronological split: first train_frac of rows (by signup_date) go to train."""
    order = np.argsort(dates.values)
    n_train = int(len(order) * train_frac)
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
    ])


def make_models(random_state: int = 42) -> dict:
    return {
        "LogisticRegression": make_lr_pipeline(random_state),
        "GradientBoosting": make_gb_pipeline(random_state),
    }
