"""Load and preprocess the churn dataset with explicit leak handling."""
import pandas as pd
from datetime import datetime


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load dataset, remove duplicates, and select features.

    Returns: (df, metadata) where metadata tracks duplicates and leak decisions.
    """
    df = pd.read_csv(csv_path)
    n_rows = len(df)

    # Dedup: exact duplicates must not straddle train/test boundary
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_rows - len(df_dedup)

    # Feature selection:
    # Include: tenure_months, monthly_spend, support_tickets (honest causal signal)
    # Exclude: customer_id (no signal), days_since_last_login (target leak: churned
    #          by definition have not logged in recently), signup_date (temporal: needs
    #          time-based split, not random)
    features = ["tenure_months", "monthly_spend", "support_tickets"]
    X = df_dedup[features].copy()
    y = df_dedup["churned"].copy()

    # For time-based split, also extract signup_date
    signup_dates = pd.to_datetime(df_dedup["signup_date"])

    metadata = {
        "n_rows_original": n_rows,
        "n_duplicates_removed": n_duplicates,
        "n_rows_clean": len(df_dedup),
        "features": features,
        "target": "churned",
        "excluded_features": {
            "customer_id": "no_signal",
            "days_since_last_login": "target_leak_churned_customers_by_definition_logged_out_recently",
            "signup_date": "temporal_feature_requires_time_split_not_random",
        },
        "class_distribution": y.value_counts().to_dict(),
    }

    return X, y, signup_dates, metadata
