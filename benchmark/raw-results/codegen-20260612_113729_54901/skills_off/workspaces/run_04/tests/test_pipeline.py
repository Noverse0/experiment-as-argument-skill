"""Pytest tests for the churn experiment pipeline."""
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Make sure src/ is importable when tests run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_and_clean, time_split, get_X_y, FEATURE_COLS, TARGET_COL
from src.models import make_lr, make_gb, MODELS
from src.evaluate import baseline_score, label_shuffle_auc, cv_score, final_test_score


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df(tmp_path_factory):
    """Generate a small dataset using make_dataset.make()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_dataset",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "make_dataset.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tmp = tmp_path_factory.mktemp("data")
    csv_path = str(tmp / "churn.csv")
    mod.make(seed=42, n=300).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def cleaned(raw_df):
    df, dedup_removed = load_and_clean(raw_df)
    return df, dedup_removed


@pytest.fixture(scope="module")
def splits(cleaned):
    df, _ = cleaned
    train, test = time_split(df)
    return train, test


@pytest.fixture(scope="module")
def arrays(splits):
    train, test = splits
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)
    return X_train, y_train, X_test, y_test


# ---- Data tests -------------------------------------------------------------

def test_dedup_removes_duplicates(cleaned):
    df, dedup_removed = cleaned
    assert dedup_removed > 0, "Expected planted duplicates to be removed"
    # After dedup, no duplicate rows on feature+target columns
    assert not df.duplicated(subset=FEATURE_COLS + [TARGET_COL]).any()


def test_no_leaky_columns(cleaned):
    df, _ = cleaned
    assert "account_status" not in df.columns, "account_status (leaky) must be dropped"


def test_time_split_no_overlap(splits):
    train, test = splits
    assert train["signup_date"].max() <= test["signup_date"].min(), \
        "Train dates must not exceed test dates (time-based split)"


def test_time_split_sizes(splits, cleaned):
    train, test = splits
    df, _ = cleaned
    total = len(train) + len(test)
    assert total == len(df)
    # ~20% in test
    assert 0.15 < len(test) / total < 0.25


def test_feature_columns_present(arrays):
    X_train, y_train, X_test, y_test = arrays
    assert X_train.shape[1] == len(FEATURE_COLS)
    assert X_test.shape[1] == len(FEATURE_COLS)


def test_target_is_binary(arrays):
    _, y_train, _, y_test = arrays
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


# ---- Model tests ------------------------------------------------------------

def test_lr_pipeline_fit_predict(arrays):
    X_train, y_train, X_test, y_test = arrays
    model = make_lr(seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_gb_pipeline_fit_predict(arrays):
    X_train, y_train, X_test, y_test = arrays
    model = make_gb(seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_models_dict_has_expected_keys():
    assert "LogisticRegression" in MODELS
    assert "GradientBoosting" in MODELS


# ---- Scaler leak test -------------------------------------------------------

def test_scaler_fitted_on_train_only(arrays):
    """Scaler mean should match train distribution, not test."""
    X_train, y_train, X_test, y_test = arrays
    model = make_lr(seed=42)
    model.fit(X_train, y_train)
    scaler = model.named_steps["scaler"]
    # Scaler mean for feature 0 (tenure_months) should be close to train mean
    train_mean = X_train[:, 0].mean()
    test_mean = X_test[:, 0].mean()
    assert abs(scaler.mean_[0] - train_mean) < 1.0, "Scaler mean doesn't match train"
    # It should NOT be the combined mean
    combined_mean = np.concatenate([X_train[:, 0], X_test[:, 0]]).mean()
    if abs(train_mean - combined_mean) > 0.5:
        assert abs(scaler.mean_[0] - combined_mean) > 0.3, \
            "Scaler appears to have been fitted on combined train+test data (leakage)"


# ---- Sanity check tests -----------------------------------------------------

def test_baseline_auc_near_half(arrays):
    X_train, y_train, X_test, y_test = arrays
    result = baseline_score(X_train, y_train, X_test, y_test)
    # Majority classifier AUC is exactly 0.5
    assert result["roc_auc"] == pytest.approx(0.5, abs=0.01)


def test_label_shuffle_auc_near_half(arrays):
    X_train, y_train, X_test, y_test = arrays
    shuffled_auc = label_shuffle_auc(make_lr, X_train, y_train, X_test, y_test, seed=42)
    # With shuffled labels and no leakage, AUC should be near 0.5
    assert shuffled_auc < 0.65, f"Label-shuffle AUC={shuffled_auc:.3f} suspiciously high"


def test_models_beat_baseline(arrays):
    X_train, y_train, X_test, y_test = arrays
    base = baseline_score(X_train, y_train, X_test, y_test)
    for name, factory in MODELS.items():
        result = final_test_score(factory, X_train, y_train, X_test, y_test)
        assert result["roc_auc"] > base["roc_auc"], \
            f"{name} AUC {result['roc_auc']:.3f} does not beat baseline {base['roc_auc']:.3f}"


def test_cv_score_shape(arrays):
    X_train, y_train, _, _ = arrays
    result = cv_score(make_lr, X_train, y_train, seeds=[42, 123])
    assert "roc_auc" in result
    assert "mean" in result["roc_auc"]
    assert "std" in result["roc_auc"]
    assert result["roc_auc"]["n"] == 2 * 5  # 2 seeds × 5 folds


def test_determinism(arrays):
    """Same seed must produce identical metrics."""
    X_train, y_train, X_test, y_test = arrays
    r1 = final_test_score(make_lr, X_train, y_train, X_test, y_test, seed=42)
    r2 = final_test_score(make_lr, X_train, y_train, X_test, y_test, seed=42)
    assert r1["roc_auc"] == pytest.approx(r2["roc_auc"], abs=1e-9)
