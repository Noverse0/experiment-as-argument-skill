"""Dataset loading and exploration."""
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load churn dataset."""
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows (excluding customer_id)."""
    cols = [c for c in df.columns if c != "customer_id"]
    return df[cols].duplicated().sum()


def get_class_balance(df: pd.DataFrame) -> dict:
    """Report target class distribution."""
    counts = df["churned"].value_counts()
    total = len(df)
    return {
        "not_churned": int(counts.get(0, 0)),
        "churned": int(counts.get(1, 0)),
        "churn_rate": float(counts.get(1, 0) / total) if total > 0 else 0.0,
    }
