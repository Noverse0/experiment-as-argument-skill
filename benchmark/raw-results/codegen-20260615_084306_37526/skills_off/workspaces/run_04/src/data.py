import pandas as pd

# days_since_last_login is excluded: it is a post-outcome feature (churned
# customers have, by definition, stopped logging in, so this value is
# recorded after the outcome is known). Including it would be target leakage.
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"
DATE_COL = "signup_date"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, drop exact duplicates."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[data] dedup: removed {dropped} exact duplicates ({before} → {len(df)} rows)")
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Sort by signup_date; earlier rows → train, later rows → test."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
