"""Shared test fixtures."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def raw_like_dataset() -> pd.DataFrame:
    """A tiny dataframe shaped like churn.csv, with planted traps.

    Includes the leak column, an exact duplicate row, and out-of-order dates
    so the preparation logic can be exercised deterministically.
    """
    rows = [
        # id, signup_date, tenure, spend, tickets, status, churned
        (1, "2023-03-01", 10, 50.0, 0, "active", 0),
        (2, "2023-01-15", 5, 80.0, 3, "closed", 1),
        (3, "2023-02-10", 20, 30.0, 1, "active", 0),
        (4, "2023-05-20", 2, 100.0, 5, "closed", 1),
        (5, "2023-04-01", 40, 20.0, 0, "active", 0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "signup_date",
            "tenure_months",
            "monthly_spend",
            "support_tickets",
            "account_status",
            "churned",
        ],
    )
    # append an exact duplicate of row 1
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)
