"""Make the src/ package importable and provide a shared prepared-data fixture."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from churn import data as data_mod  # noqa: E402

DATA_PATH = ROOT / "churn.csv"


@pytest.fixture(scope="session")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} missing; run: python3 make_dataset.py --out churn.csv")
    return data_mod.load_raw(str(DATA_PATH))


@pytest.fixture(scope="session")
def prepared(raw_df):
    return data_mod.prepare(raw_df)
