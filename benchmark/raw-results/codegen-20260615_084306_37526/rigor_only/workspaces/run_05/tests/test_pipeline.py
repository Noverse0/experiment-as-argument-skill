"""Pipeline correctness and rigor tests."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.pipeline import (
    FEATURE_COLS,
    LEAKY_COLS,
    deduplicate,
    engineer_features,
    get_X_y,
    make_gb_pipeline,
    make_lr_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Random features with random labels — use only for structural tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "days_since_last_login": rng.integers(0, 100, n),
        "churned": rng.integers(0, 2, n),
    })
    return df


def _make_df_with_signal(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same causal structure as the real dataset."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d")
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": list(signup[:n]),
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "days_since_last_login": rng.integers(0, 100, n),
        "churned": churn,
    })


@pytest.fixture
def sample_df():
    return _make_df(n=200)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

def test_deduplicate_removes_exact_duplicates():
    df = _make_df(n=100)
    dup = df.sample(n=20, random_state=0)
    combined = pd.concat([df, dup], ignore_index=True)
    assert len(combined) == 120
    deduped = deduplicate(combined)
    assert len(deduped) == 100


def test_deduplicate_no_duplicates_unchanged():
    df = _make_df(n=50)
    assert len(deduplicate(df)) == len(df)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

def test_engineer_features_adds_days_since_signup(sample_df):
    result = engineer_features(sample_df)
    assert "days_since_signup" in result.columns
    assert result["days_since_signup"].dtype in [np.dtype("int64"), np.dtype("float64")]


def test_engineer_features_non_negative_days(sample_df):
    result = engineer_features(sample_df)
    assert (result["days_since_signup"] >= 0).all()


def test_engineer_features_original_unchanged(sample_df):
    engineer_features(sample_df)
    assert "days_since_signup" not in sample_df.columns


# ---------------------------------------------------------------------------
# Leak exclusion tests
# ---------------------------------------------------------------------------

def test_leaky_column_not_in_features(sample_df):
    X, _ = get_X_y(sample_df)
    for col in LEAKY_COLS:
        assert col not in X.columns, f"Leaked column '{col}' found in features"


def test_customer_id_not_in_features(sample_df):
    X, _ = get_X_y(sample_df)
    assert "customer_id" not in X.columns


def test_only_expected_feature_cols_present(sample_df):
    X, _ = get_X_y(sample_df)
    assert list(X.columns) == FEATURE_COLS


def test_target_not_in_features(sample_df):
    X, _ = get_X_y(sample_df)
    assert "churned" not in X.columns


# ---------------------------------------------------------------------------
# Pipeline reproducibility
# ---------------------------------------------------------------------------

def test_lr_pipeline_reproducible(sample_df):
    X, y = get_X_y(sample_df)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    scores_a = cross_val_score(make_lr_pipeline(42), X, y, cv=cv, scoring="roc_auc")
    scores_b = cross_val_score(make_lr_pipeline(42), X, y, cv=cv, scoring="roc_auc")
    np.testing.assert_array_almost_equal(scores_a, scores_b)


def test_gb_pipeline_reproducible(sample_df):
    X, y = get_X_y(sample_df)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    scores_a = cross_val_score(make_gb_pipeline(42), X, y, cv=cv, scoring="roc_auc")
    scores_b = cross_val_score(make_gb_pipeline(42), X, y, cv=cv, scoring="roc_auc")
    np.testing.assert_array_almost_equal(scores_a, scores_b)


# ---------------------------------------------------------------------------
# Label-shuffle sanity check
# ---------------------------------------------------------------------------

def test_label_shuffle_degrades_gb_performance():
    """With shuffled labels, AUC must fall toward baseline (≈ 0.5)."""
    df = _make_df_with_signal(n=400, seed=42)
    X, y = get_X_y(df)
    rng = np.random.default_rng(99)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    real_auc = float(np.mean(cross_val_score(make_gb_pipeline(), X, y, cv=cv, scoring="roc_auc")))
    shuffle_auc = float(np.mean(cross_val_score(make_gb_pipeline(), X, y_shuffled, cv=cv, scoring="roc_auc")))

    assert shuffle_auc < real_auc - 0.05, (
        f"Shuffled AUC {shuffle_auc:.3f} not much lower than real AUC {real_auc:.3f}; "
        "pipeline may be leaking information"
    )


# ---------------------------------------------------------------------------
# Scaler fit-on-train-only test
# ---------------------------------------------------------------------------

def test_scaler_fitted_only_on_train_fold(sample_df):
    """StandardScaler inside Pipeline must not see test fold data."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = get_X_y(sample_df)
    pipe = make_lr_pipeline()

    train_idx = X.index[:160]
    test_idx = X.index[160:]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train = y.loc[train_idx]

    pipe.fit(X_train, y_train)
    scaler: StandardScaler = pipe.named_steps["scaler"]

    # Scaler statistics should reflect train mean only
    train_mean = X_train.mean().values
    np.testing.assert_allclose(scaler.mean_, train_mean, rtol=1e-5)
