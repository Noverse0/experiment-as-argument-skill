"""Tests for the data discipline: dedup, leak exclusion, chronological split."""
import pandas as pd

from churn_experiment import config
from churn_experiment.data import (
    audit,
    chronological_split,
    deduplicate,
    to_xy,
)


def test_audit_detects_duplicates_and_leak(raw_df):
    a = audit(raw_df)
    assert a.n_duplicate_rows == 200, "the planted 200 duplicates must be detected"
    assert a.account_status_is_leak is True, "account_status is a perfect target leak"
    assert 0.2 < a.churn_rate < 0.35


def test_deduplicate_removes_exact_duplicates(raw_df):
    deduped = deduplicate(raw_df)
    assert deduped.duplicated().sum() == 0
    assert len(deduped) == len(raw_df) - 200


def test_chronological_split_has_no_time_overlap(raw_df):
    deduped = deduplicate(raw_df)
    train_df, test_df = chronological_split(deduped)
    # Every test row must be dated >= the last train row (forward-looking).
    assert train_df[config.TIME_COL].max() <= test_df[config.TIME_COL].min()
    # Sizes roughly match the configured fraction.
    frac = len(test_df) / (len(train_df) + len(test_df))
    assert abs(frac - config.TEST_FRACTION) < 0.02


def test_dedup_before_split_prevents_straddling(raw_df):
    deduped = deduplicate(raw_df)
    train_df, test_df = chronological_split(deduped)
    # No row content appears on both sides of the boundary.
    merged = pd.merge(train_df, test_df, how="inner")
    assert len(merged) == 0


def test_feature_matrix_excludes_leak_and_id(raw_df):
    X, y = to_xy(raw_df)
    assert list(X.columns) == config.FEATURES
    for forbidden in config.LEAK_COLS + config.ID_COLS + [config.TIME_COL]:
        assert forbidden not in X.columns, f"{forbidden} must never reach the model"
    assert set(y.unique()).issubset({0, 1})
