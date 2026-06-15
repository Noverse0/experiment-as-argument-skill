"""Data loading, deduplication, and feature selection for the churn experiment."""
import pandas as pd

FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"
LEAKY_COLS = ["days_since_last_login"]


def load_and_prepare(path: str):
    """
    Load churn CSV, drop exact duplicates, drop post-outcome leaky feature.

    days_since_last_login is a post-outcome measurement: churned customers stop
    logging in, so this value is recorded *after* the churn event and is
    statistically derived from the label. Including it would inflate AUC by
    ~0.35 points (confirmed by sanity check in run_experiment.py).

    Duplicate rows (200 planted) are dropped before any split so they cannot
    straddle the train/test boundary and inflate test scores.

    Returns
    -------
    X           : DataFrame with FEATURES columns only
    y           : Series of binary churn labels
    X_audit     : DataFrame with FEATURES + LEAKY_COLS (for leak audit only)
    stats       : dict with dataset provenance counts and churn rate
    """
    df = pd.read_csv(path)
    n_raw = len(df)
    churn_rate_raw = float(df[TARGET].mean())

    dedup_subset = FEATURES + LEAKY_COLS + [TARGET]
    df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    n_dupes_dropped = n_raw - len(df)
    n_clean = len(df)
    churn_rate_clean = float(df[TARGET].mean())

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_audit = df[FEATURES + LEAKY_COLS].copy()

    stats = {
        "n_raw": n_raw,
        "n_dupes_dropped": n_dupes_dropped,
        "n_clean": n_clean,
        "churn_rate_raw": churn_rate_raw,
        "churn_rate_clean": churn_rate_clean,
    }
    return X, y, X_audit, stats
