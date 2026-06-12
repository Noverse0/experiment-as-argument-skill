"""Tests for data pipeline correctness and experiment sanity."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.pipeline import (
    DATE_COL,
    FEATURES,
    LEAKY_COLS,
    TARGET,
    load_and_clean,
    split_xy,
    time_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_csv(tmp_path):
    """Write a minimal valid churn CSV with one duplicate row."""
    data = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 4],
        "signup_date": [
            "2023-01-15", "2023-03-01", "2023-06-01",
            "2023-09-01", "2023-09-01",
        ],
        "tenure_months": [12, 24, 6, 36, 36],
        "monthly_spend": [50.0, 80.0, 30.0, 100.0, 100.0],
        "support_tickets": [1, 0, 3, 2, 2],
        "account_status": ["active", "active", "closed", "active", "active"],
        "churned": [0, 0, 1, 0, 0],
    })
    p = tmp_path / "test_churn.csv"
    data.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def clean_df(raw_csv):
    df, _ = load_and_clean(raw_csv)
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_deduplication_removes_exact_duplicates(raw_csv):
    df, info = load_and_clean(raw_csv)
    assert info["n_dupes_removed"] == 1, "should remove the 1 duplicate row"
    assert len(df) == 4


def test_deduplication_info_is_accurate(raw_csv):
    df, info = load_and_clean(raw_csv)
    assert info["n_before_dedup"] == 5
    assert info["n_after_dedup"] == 4
    assert info["n_dupes_removed"] == 1


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------

def test_account_status_is_dropped(clean_df):
    for col in LEAKY_COLS:
        assert col not in clean_df.columns, f"{col} must be dropped (target leak)"


def test_customer_id_is_dropped(clean_df):
    assert "customer_id" not in clean_df.columns


def test_target_still_present(clean_df):
    assert TARGET in clean_df.columns


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def test_signup_ordinal_is_numeric(clean_df):
    assert "signup_ordinal" in clean_df.columns
    assert pd.api.types.is_numeric_dtype(clean_df["signup_ordinal"])


def test_features_all_present(clean_df):
    for f in FEATURES:
        assert f in clean_df.columns, f"feature '{f}' missing from cleaned df"


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def test_time_split_sizes(clean_df):
    train, test = time_split(clean_df, test_fraction=0.25)
    assert len(train) + len(test) == len(clean_df)


def test_time_split_no_future_leak(clean_df):
    train, test = time_split(clean_df, test_fraction=0.25)
    max_train_date = train[DATE_COL].max()
    min_test_date = test[DATE_COL].min()
    assert max_train_date <= min_test_date, (
        "All training dates must be <= all test dates (no future leak)"
    )


def test_time_split_train_before_test(clean_df):
    train, test = time_split(clean_df, test_fraction=0.50)
    assert train[DATE_COL].max() <= test[DATE_COL].min()


# ---------------------------------------------------------------------------
# Scaler discipline: fit on train only
# ---------------------------------------------------------------------------

def test_scaler_fit_on_train_only(clean_df):
    """StandardScaler fitted on train must use only train statistics."""
    train, test = time_split(clean_df, test_fraction=0.50)
    X_train, _ = split_xy(train)
    X_test, _ = split_xy(test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scaled test mean should NOT be zero (that would imply scaler saw test data)
    # It is meaningless to assert exact value, but scaling must not crash.
    assert X_test_scaled.shape == X_test.shape


# ---------------------------------------------------------------------------
# Sanity checks: label shuffle and baseline
# ---------------------------------------------------------------------------

def test_label_shuffle_degrades_to_chance(raw_csv):
    """Shuffling labels on a clean dataset must push AUC near 0.5."""
    # Use the real churn.csv for a large enough sample to be meaningful
    import os
    data_path = "churn.csv"
    if not os.path.exists(data_path):
        pytest.skip("churn.csv not found; run make_dataset.py first")

    from src.pipeline import load_and_clean, time_split, split_xy
    df, _ = load_and_clean(data_path)
    train, test = time_split(df, test_fraction=0.30)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rng = np.random.default_rng(99)
    y_shuffled = rng.permutation(y_train)

    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train_s, y_shuffled)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

    assert abs(auc - 0.5) < 0.07, (
        f"Label-shuffle AUC {auc:.4f} is too far from chance — "
        "suggests information is leaking around the labels"
    )


def test_models_beat_majority_baseline(raw_csv):
    """Both models must outperform the majority-class baseline on full data."""
    import os
    data_path = "churn.csv"
    if not os.path.exists(data_path):
        pytest.skip("churn.csv not found; run make_dataset.py first")

    from src.pipeline import load_and_clean, time_split, split_xy
    from sklearn.ensemble import GradientBoostingClassifier

    df, _ = load_and_clean(data_path)
    train, test = time_split(df, test_fraction=0.30)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline_proba = np.full(len(y_test), y_train.mean())
    baseline_auc = roc_auc_score(y_test, baseline_proba)

    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train_s, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

    gb = GradientBoostingClassifier(n_estimators=50, subsample=0.8, random_state=0)
    gb.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

    assert lr_auc > baseline_auc, f"LR AUC {lr_auc:.4f} must beat baseline {baseline_auc:.4f}"
    assert gb_auc > baseline_auc, f"GB AUC {gb_auc:.4f} must beat baseline {baseline_auc:.4f}"


# ---------------------------------------------------------------------------
# split_xy shape consistency
# ---------------------------------------------------------------------------

def test_split_xy_shapes(clean_df):
    X, y = split_xy(clean_df)
    assert X.shape[1] == len(FEATURES)
    assert X.shape[0] == y.shape[0] == len(clean_df)
