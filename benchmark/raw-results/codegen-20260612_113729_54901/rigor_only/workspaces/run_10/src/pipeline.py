"""Data loading, feature engineering, and model pipeline factories."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Columns that encode the target and must never enter the feature matrix
LEAKAGE_COLS = ["account_status"]
ID_COLS = ["customer_id"]
TARGET = "churned"
DATE_COL = "signup_date"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, and remove exact duplicate rows."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"Deduplication: removed {removed} exact duplicate rows ({before} → {len(df)})")
    return df


def make_features(df: pd.DataFrame):
    """Return (X, y) after dropping leakage, ID, and target columns.

    signup_date is converted to signup_age_days (days since the earliest
    signup in the provided dataframe) so downstream models see a numeric value
    that preserves temporal ordering without raw date leakage.
    """
    df = df.copy()

    # Convert date to numeric; pd.to_datetime is idempotent if already parsed
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    min_date = df[DATE_COL].min()
    df["signup_age_days"] = (df[DATE_COL] - min_date).dt.days

    drop_cols = LEAKAGE_COLS + ID_COLS + [TARGET, DATE_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y


def make_lr_pipeline() -> Pipeline:
    """Logistic Regression with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def make_gb_pipeline() -> Pipeline:
    """Gradient Boosting — no scaling required."""
    return Pipeline([
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])
