"""Tests for leak-aware data preparation."""
from churn_experiment import data as data_mod


def test_leak_and_id_columns_dropped(raw_like_dataset):
    prepared = data_mod.prepare(raw_like_dataset)
    assert list(prepared.X.columns) == data_mod.FEATURE_COLUMNS
    # the perfect-leak column and id never reach the feature matrix
    assert "account_status" not in prepared.X.columns
    assert "customer_id" not in prepared.X.columns
    assert "signup_date" not in prepared.X.columns


def test_exact_duplicates_removed_before_split(raw_like_dataset):
    prepared = data_mod.prepare(raw_like_dataset)
    # one duplicate planted by the fixture
    assert prepared.n_duplicates_dropped == 1
    assert prepared.n_raw == len(raw_like_dataset)
    assert len(prepared.X) == prepared.n_raw - prepared.n_duplicates_dropped


def test_rows_sorted_ascending_by_time(raw_like_dataset):
    prepared = data_mod.prepare(raw_like_dataset)
    times = prepared.time.tolist()
    assert times == sorted(times)


def test_features_and_target_aligned(raw_like_dataset):
    prepared = data_mod.prepare(raw_like_dataset)
    assert len(prepared.X) == len(prepared.y) == len(prepared.time)
    assert set(prepared.y.unique()).issubset({0, 1})


def test_real_dataset_dedup_and_leak(tmp_path):
    """On the real generated file: 200 dups dropped, leak column gone."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "churn.csv"
    subprocess.run(
        [sys.executable, str(root / "make_dataset.py"), "--out", str(out)],
        check=True,
    )
    prepared = data_mod.prepare(data_mod.load_raw(str(out)))
    assert prepared.n_duplicates_dropped == 200
    assert "account_status" not in prepared.X.columns
