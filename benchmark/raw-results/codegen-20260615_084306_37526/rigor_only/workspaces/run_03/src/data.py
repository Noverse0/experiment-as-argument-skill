import pandas as pd

# Only legitimate causal features: observable before the churn outcome is known.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"

# Dropped columns and reasons:
#   days_since_last_login — target leak: churned customers stop logging in, so
#     this value is recorded at/after the outcome. The column name sounds benign
#     but the value is causally downstream of churn, not upstream.
#   signup_date — temporal column; tenure_months already captures time-on-platform
#     and is cleaner. Including raw dates would require careful encoding and adds
#     no information not already in tenure_months.
#   customer_id — row identifier with no predictive signal.
DROPPED = ["customer_id", "signup_date", "days_since_last_login"]


def load(path: str) -> tuple:
    """Load CSV, deduplicate, and return (X, y)."""
    df = pd.read_csv(path)
    # 200 exact duplicate rows were appended in the data generator; remove before
    # any split so identical rows cannot straddle train/test.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"[data] removed {n_dupes} exact duplicate rows ({n_before} → {len(df)})")
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
