"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset and remove duplicates.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with duplicates removed.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Check for exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Removed {n_before - n_after} exact duplicate rows")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target, remove identifiers and leaky columns.

    Data discipline:
    - Drop customer_id (identifier)
    - Drop days_since_last_login (target leak: recorded post-outcome)
    - Drop signup_date (temporal, not used for this comparison)
    - Use: tenure_months, monthly_spend, support_tickets

    Args:
        df: Raw dataframe with all columns.

    Returns:
        (features DataFrame, target Series)
    """
    # Target
    y = df["churned"].copy()

    # Features: safe columns only
    # Excluded:
    # - customer_id: just an index
    # - days_since_last_login: target leak (value recorded after churn)
    # - signup_date: temporal; not forward-looking for this task
    X = df[["tenure_months", "monthly_spend", "support_tickets"]].copy()

    print(f"Feature count: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


class FitOnTrainScaler:
    """Wrapper to ensure scaling is fit on train only, applied to all splits."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train: pd.DataFrame) -> "FitOnTrainScaler":
        """Fit scaler on training data only."""
        self.scaler.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
