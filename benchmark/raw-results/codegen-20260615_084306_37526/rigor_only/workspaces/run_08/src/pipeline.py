"""Data loading, cleaning, and model construction for the churn experiment."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"
# Columns intentionally excluded and why (documented here for audit trail):
#   customer_id        — row identifier, zero predictive signal
#   signup_date        — temporal; no deployment time anchor defined, random split invalid
#   days_since_last_login — TARGET LEAK: recorded after churn outcome is known
#                          (churned customers stop logging in, so value encodes the label)
DROP_COLS = ["customer_id", "signup_date", "days_since_last_login"]


def load_and_clean(path: str) -> tuple:
    df = pd.read_csv(path)
    before = len(df)
    # Deduplicate before any split; 200 exact duplicate rows are planted in the dataset.
    # Duplicates straddling train/test would inflate test metrics.
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    dropped = before - after

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y, dropped


def make_models() -> dict:
    """Return named sklearn Pipelines. StandardScaler applied to both for a fair comparison."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ])
    return {"LogisticRegression": lr, "GradientBoosting": gb}
