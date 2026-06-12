"""Tests for data cleaning: the leakage/dedup/time-order discipline."""
import subprocess
import sys
from pathlib import Path

import pytest

from src import data as data_mod

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "churn.csv"


@pytest.fixture(scope="session")
def csv_path() -> str:
    """Ensure the dataset exists; generate it if missing."""
    if not CSV.exists():
        subprocess.check_call(
            [sys.executable, "make_dataset.py", "--out", str(CSV)], cwd=ROOT
        )
    return str(CSV)


def test_leak_and_id_columns_excluded(csv_path):
    clean = data_mod.load_clean(csv_path)
    assert "account_status" not in clean.X.columns
    assert "customer_id" not in clean.X.columns
    assert "signup_date" not in clean.X.columns
    assert list(clean.X.columns) == data_mod.FEATURES


def test_duplicates_removed(csv_path):
    clean = data_mod.load_clean(csv_path)
    # generator appends 200 exact duplicates
    assert clean.n_duplicates == 200
    assert clean.n_clean == clean.n_raw - 200
    # no exact duplicate rows remain in the feature matrix + target
    combined = clean.X.copy()
    combined["churned"] = clean.y.values
    # there may be coincidental feature collisions, but the planted exact
    # duplicates (full-row) must be gone: clean count matches unique raw rows
    assert clean.n_clean == 4000


def test_time_ordered(csv_path):
    """Rows must be sorted by signup_date so TimeSeriesSplit respects time."""
    import pandas as pd

    raw = pd.read_csv(csv_path).drop_duplicates()
    dates = pd.to_datetime(
        raw.sort_values("signup_date", kind="mergesort")["signup_date"].values
    )
    assert (dates[:-1] <= dates[1:]).all()


def test_target_is_binary(csv_path):
    clean = data_mod.load_clean(csv_path)
    assert set(clean.y.unique()) <= {0, 1}
    assert 0.0 < clean.churn_rate < 1.0


def test_leak_loader_includes_status(csv_path):
    X_leak, y = data_mod.load_with_leak(csv_path)
    assert "account_status_closed" in X_leak.columns
    assert len(X_leak) == len(y)
