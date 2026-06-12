"""Shared fixtures. Builds a small churn CSV via the project's own generator so
tests exercise the real planted traps (leak column, duplicates, temporal dates)
without depending on a checked-in data file."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_make_dataset():
    spec = importlib.util.spec_from_file_location(
        "make_dataset", ROOT / "make_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def churn_csv(tmp_path_factory) -> str:
    """A small, deterministic churn CSV with the same trap structure as the real one."""
    make_dataset = _load_make_dataset()
    df = make_dataset.make(seed=7, n=600)  # smaller for speed; still has dupes + leak
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    df.to_csv(out, index=False)
    return str(out)
