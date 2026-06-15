"""Feature engineering and model pipeline factories.

days_since_last_login is excluded: it is recorded after the churn outcome
(churned customers have, by definition, stopped logging in), making it a
target leak rather than a predictive feature available at decision time.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

REFERENCE_DATE = pd.Timestamp("2023-01-01")

# Leaked post-hoc column: value is determined by the outcome, not available
# before the outcome is known.
LEAKY_COLS = ["days_since_last_login"]

# Non-predictive identifiers and raw temporal column replaced by engineered feature.
DROP_COLS = ["customer_id", "signup_date"]

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "days_since_signup"]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Must run before any split."""
    return df.drop_duplicates().reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_since_signup from signup_date."""
    df = df.copy()
    df["days_since_signup"] = (
        pd.to_datetime(df["signup_date"]) - REFERENCE_DATE
    ).dt.days
    return df


def get_X_y(df: pd.DataFrame):
    """Return feature matrix and target after engineering and dropping leaks."""
    df = engineer_features(df)
    X = df[FEATURE_COLS].copy()
    y = df["churned"].copy()
    return X, y


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    """Logistic regression with standard scaling (fit on train only in CV)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    """Gradient boosting; tree-based models do not need scaling."""
    return Pipeline([
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )),
    ])
