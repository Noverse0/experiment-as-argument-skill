"""Data loading, deduplication, and splitting for the churn experiment."""
import pandas as pd

# Only legitimate causal features survive the leak audit.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"

# days_since_last_login is a target leak: it is recorded *after* churn occurs
# (a churned customer has already stopped logging in), so it encodes the outcome.
# Including it would inflate AUC without measuring genuine predictive signal.
LEAK_COLS = ["days_since_last_login"]


def load_data(path: str) -> pd.DataFrame:
    """Read CSV, remove the 200 appended duplicate rows, return clean frame."""
    df = pd.read_csv(path)
    # The generator appended 200 rows sampled from the original 4000; those rows
    # share customer_id values with originals, so drop_duplicates on customer_id
    # removes them while preserving unique customers.
    before = len(df)
    df = df.drop_duplicates(subset="customer_id").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  dedup: removed {removed} duplicate rows ({before} → {len(df)})")
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Sort by signup_date, cut at train_frac. No time leak across boundary."""
    df = df.sort_values("signup_date").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def get_xy(df: pd.DataFrame):
    """Return (X, y) using only the clean feature set."""
    return df[FEATURES].copy(), df[TARGET].copy()
