"""Data pipeline: load, deduplicate, split, preprocess."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and remove exact duplicates (except first occurrence)."""
    df = pd.read_csv(csv_path)
    initial_len = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dupes = initial_len - len(df)
    print(f"Removed {n_dupes} exact duplicate rows.")
    return df


def check_target_balance(y: np.ndarray) -> float:
    """Return positive class rate."""
    return y.mean()


def time_based_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split by signup_date (older→train, newer→test) to respect temporal order.

    Returns:
        (df_train, df_test) DataFrames.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select features, excluding: customer_id (index), signup_date (temporal var),
    account_status (leaked from target).
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    return df[feature_cols].copy()


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale features: fit scaler on train, apply to both."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_time_series_splits(csv_path: str, n_splits: int = 3) -> list:
    """Generate time-based K-fold splits using TimeSeriesSplit.

    Each fold uses progressively more data for training while keeping
    the test set in future time. This respects temporal order.

    Args:
        csv_path: path to CSV.
        n_splits: number of folds.

    Yields:
        dict with X_train, X_test, y_train, y_test, target_rate, fold_idx.
    """
    df = load_and_clean(csv_path)
    df = df.sort_values("signup_date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_full = select_features(df)
    y_full = df["churned"].values

    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_full)):
        X_train = X_full.iloc[train_idx]
        y_train = y_full[train_idx]

        X_test = X_full.iloc[test_idx]
        y_test = y_full[test_idx]

        X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)
        target_rate = check_target_balance(y_train)

        splits.append({
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "target_rate": target_rate,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "fold_idx": fold_idx,
        })

    return splits


def prepare_split(csv_path: str, test_size: float = 0.2,
                  seed: int = None) -> dict:
    """Full pipeline: load, deduplicate, split, preprocess.

    Args:
        csv_path: path to CSV.
        test_size: fraction for test set.
        seed: ignored (time-based split is deterministic).

    Returns:
        dict with X_train, X_test, y_train, y_test, target_rate, n_duplicates.
    """
    df = load_and_clean(csv_path)
    df_train, df_test = time_based_split(df, test_size=test_size)

    X_train = select_features(df_train)
    y_train = df_train["churned"].values

    X_test = select_features(df_test)
    y_test = df_test["churned"].values

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    target_rate = check_target_balance(y_train)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "target_rate": target_rate,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
