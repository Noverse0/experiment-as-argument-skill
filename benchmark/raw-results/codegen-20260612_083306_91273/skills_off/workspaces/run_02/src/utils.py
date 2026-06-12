"""Data loading and preprocessing for churn experiment."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and remove known leak: account_status is derived from churned."""
    df = pd.read_csv(csv_path)

    # Drop account_status: it is perfectly determined by churned (leak).
    # Documented in make_dataset.py: "closed" iff churned=1.
    df = df.drop(columns=["account_status"])

    # Drop customer_id; not a predictive feature.
    df = df.drop(columns=["customer_id"])

    return df


def deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates to prevent them straddling train/test boundary.

    make_dataset.py plants 200 duplicates. With random split, these can straddle
    the boundary. Deduplication ensures all information in test is truly held out.
    """
    initial_rows = len(df)
    df = df.drop_duplicates(keep="first")
    removed = initial_rows - len(df)
    print(f"[Data] Removed {removed} duplicate rows; {len(df)} rows remain")
    return df


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7,
                     date_col: str = "signup_date") -> tuple:
    """Split data by signup date (time-based), not randomly.

    This prevents leakage from temporal information. Models trained on past
    customers predict future churn.
    """
    df = df.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"[Data] Time-based split: {len(train)} train, {len(test)} test")
    print(f"[Data] Train period: {train[date_col].min()} to {train[date_col].max()}")
    print(f"[Data] Test period:  {test[date_col].min()} to {test[date_col].max()}")

    return train, test


def preprocess_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       date_col: str = "signup_date") -> tuple:
    """Fit preprocessing on train, apply to both. Drop date column after use.

    Returns (X_train, X_test, y_train, y_test, scaler).
    """
    # Drop date column (not a learnable feature for this task).
    train_clean = train_df.drop(columns=[date_col])
    test_clean = test_df.drop(columns=[date_col])

    # Separate target.
    y_train = train_clean["churned"].values
    y_test = test_clean["churned"].values

    X_train = train_clean.drop(columns=["churned"])
    X_test = test_clean.drop(columns=["churned"])

    # Fit scaler on train only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[Preprocess] Target rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")
    print(f"[Preprocess] Features: {X_train.columns.tolist()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def baseline_predictions(y_test: np.ndarray) -> float:
    """Majority class baseline: always predict the most common class."""
    majority_class = np.bincount(y_test).argmax()
    baseline_pred = np.full_like(y_test, majority_class, dtype=float)
    # Accuracy of baseline
    acc = (baseline_pred == y_test).mean()
    return acc
