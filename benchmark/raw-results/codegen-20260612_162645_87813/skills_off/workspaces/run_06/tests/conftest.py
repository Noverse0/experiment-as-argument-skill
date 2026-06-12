"""Make the src/ package importable and provide a shared dataset fixture."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from churn_experiment.data import load_raw  # noqa: E402


@pytest.fixture(scope="session")
def churn_csv():
    """Generate a temporary dataset once for the whole test session."""
    import subprocess
    import tempfile

    d = tempfile.mkdtemp()
    out = os.path.join(d, "churn.csv")
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_dataset.py"), "--out", out],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def raw_df(churn_csv):
    return load_raw(churn_csv)
