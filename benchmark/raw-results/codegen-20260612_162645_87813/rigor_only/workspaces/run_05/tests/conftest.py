"""Shared fixtures: generate a small churn CSV once per test session."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory) -> str:
    """Generate a dataset via the project's make_dataset.py into a temp file."""
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "make_dataset.py"), "--out", str(out)],
        check=True,
        cwd=ROOT,
    )
    return str(out)
