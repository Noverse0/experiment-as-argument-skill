"""Data loading, deduplication, temporal splitting, and pipeline construction.

Design decisions:
- Drop account_status: it encodes the target directly ("closed" iff churned==1).
- Drop customer_id: row identifier with no predictive value.
- Deduplicate before splitting: 200 exact duplicate rows exist; if they straddle the
  split the model can memorise train rows and look them up at test time.
- Time-based split: signup_date is temporal. Random splits would mix future customers
  into training data, inflating apparent generalisation.
- signup_date converted to ordinal days as a numeric feature after dedup/split.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "churned"
DROP_COLS = ["customer_id", "account_status", "signup_date"]
DATE_COL = "signup_date"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    # Deduplicate before any split to prevent train/test contamination.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate rows before splitting.")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by signup_date and take the last test_frac as test set."""
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert signup_date to numeric days-since-epoch, drop non-feature columns."""
    out = df.copy()
    out["signup_day"] = (out[DATE_COL] - pd.Timestamp("1970-01-01")).dt.days
    return out.drop(columns=DROP_COLS)


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat = featurise(df)
    X = feat.drop(columns=[TARGET])
    y = feat[TARGET]
    return X, y


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
        )),
    ])
