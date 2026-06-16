import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(csv_path):
    """Load churn dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def validate_data(df):
    """Check for anomalies in the dataset."""
    assert df.shape[0] > 0, "Dataset is empty"
    assert "churned" in df.columns, "Target column 'churned' not found"
    assert df["churned"].notna().all(), "Target has missing values"
    # Check for exact duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"Warning: {dupes} exact duplicate rows found")
    return True


def preprocess_and_split(df, test_size=0.3, random_state=None):
    """
    Split first, then preprocess.
    Returns: X_train, X_test, y_train, y_test, scaler

    Features used: tenure_months, monthly_spend, support_tickets
    Dropped: customer_id (no signal), signup_date (use tenure instead),
             days_since_last_login (LEAK: encodes churn via timing test)
    """
    features = ["tenure_months", "monthly_spend", "support_tickets"]
    X = df[features].copy()
    y = df["churned"].copy()

    # Split before any fit-like operation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_baseline_rate(y):
    """Return the majority class accuracy (baseline floor)."""
    return max(y.value_counts()) / len(y)
