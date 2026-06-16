"""Data loading and preprocessing utilities."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and apply data discipline rules.

    Rules:
    - Drop target leak: days_since_last_login (churned customers have high values by definition)
    - Deduplicate exact rows before any split (prevents train/test contamination)
    - Drop signup_date (temporal column, not used for this analysis)
    """
    df = pd.read_csv(csv_path)

    initial_rows = len(df)

    # Drop signup_date: temporal column; using random split, so not useful
    df = df.drop(columns=["signup_date"])

    # Deduplicate exact rows BEFORE split
    # This is critical: the dataset has 200 appended duplicates that could straddle train/test
    df = df.drop_duplicates()
    dedup_rows = initial_rows - len(df)

    # Drop customer_id (not a feature)
    df = df.drop(columns=["customer_id"])

    # Drop days_since_last_login: TARGET LEAK
    # Rationale: A churned customer has by definition stopped logging in.
    # This value is recorded at/after the outcome, not at prediction time.
    # It's a disguised leak with a plausible name, but causally impossible to know pre-outcome.
    if "days_since_last_login" in df.columns:
        df = df.drop(columns=["days_since_last_login"])

    print(f"Data cleaning: removed {dedup_rows} duplicate rows, kept {len(df)} rows")

    return df


def split_and_preprocess(df: pd.DataFrame, test_size: float = 0.3, random_state: int = None):
    """Split data and apply preprocessing.

    Rules:
    - Split before transform: stratified split on target
    - Fit scaler on train only, apply to both
    """
    X = df.drop(columns=["churned"])
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return data summary statistics."""
    y = df["churned"]
    return {
        "total_rows": len(df),
        "n_features": len(df.columns) - 1,
        "target_rate": y.mean(),
        "n_positive": y.sum(),
        "n_negative": (1 - y).sum(),
    }
