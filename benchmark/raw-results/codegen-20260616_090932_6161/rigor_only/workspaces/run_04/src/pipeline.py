"""Data loading, preprocessing, and sanity checks."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    """Load churn dataset."""
    df = pd.read_csv(path)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Returns deduplicated dataframe."""
    initial_rows = len(df)
    df_dedup = df.drop_duplicates(keep='first')
    removed = initial_rows - len(df_dedup)
    print(f"Deduplication: removed {removed} exact duplicates")
    return df_dedup.reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess: remove target leak, engineer features.

    Removed: days_since_last_login (target leak — churned customers have high
    values *because* they churned, recorded post-outcome).
    """
    df = df.copy()
    # Days since signup (time-based feature, safe).
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['days_since_signup'] = (df['signup_date'].max() - df['signup_date']).dt.days

    # Drop unsafe columns.
    df = df.drop(columns=['customer_id', 'signup_date', 'days_since_last_login'])

    return df


def time_based_split(
    df: pd.DataFrame,
    target_col: str = 'churned',
    train_frac: float = 0.7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based split (respect temporal order for signup_date).
    Since we dropped signup_date, use days_since_signup as proxy.
    Returns: X_train, X_test, y_train, y_test
    """
    df = df.sort_values('days_since_signup', ascending=True).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Train set: {len(X_train)} rows, churn rate: {y_train.mean():.3f}")
    print(f"Test set: {len(X_test)} rows, churn rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale features: fit scaler on train only, apply to both.
    Returns: X_train_scaled, X_test_scaled (numpy arrays)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def get_churn_rate(y: pd.Series) -> float:
    """Get target churn rate."""
    return y.mean()


def sanity_check_baseline(y_test: pd.Series) -> float:
    """
    Baseline: majority class. Models must beat this.
    Returns: baseline accuracy.
    """
    baseline_pred = (y_test == 1).astype(int)  # Predict 1 (churn) always
    # Actually, baseline should be majority class. Let's check:
    churn_rate = y_test.mean()
    # Majority baseline is predicting the most common class.
    if churn_rate >= 0.5:
        baseline_pred = np.ones(len(y_test))
    else:
        baseline_pred = np.zeros(len(y_test))

    baseline_acc = (baseline_pred == y_test.values).mean()
    print(f"Baseline (majority class) accuracy: {baseline_acc:.4f}")
    return baseline_acc


def sanity_check_label_shuffle(
    model,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    seed: int = 42,
) -> float:
    """
    Label shuffle test: fit model on shuffled labels, check metric is at baseline.
    This guards against leakage — the model should not perform better than random.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y_train_shuffled = rng.permutation(y_train.values)

    model_shuffle = model.__class__(**model.get_params())
    model_shuffle.fit(X_train, y_train_shuffled)
    y_pred_shuffle = model_shuffle.predict_proba(X_test)[:, 1]

    auc_shuffle = roc_auc_score(y_test, y_pred_shuffle)
    print(f"Label shuffle AUC: {auc_shuffle:.4f} (should be ~0.5)")

    return auc_shuffle
