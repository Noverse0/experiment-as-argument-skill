"""Shared fixtures: a small generated churn dataset, made once per session."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_dataset import make  # noqa: E402


@pytest.fixture(scope="session")
def df_raw():
    # Smaller n keeps tests fast but preserves the planted traps
    # (account_status leak, 200 duplicates, temporal signup_date).
    return make(seed=7, n=1500)
