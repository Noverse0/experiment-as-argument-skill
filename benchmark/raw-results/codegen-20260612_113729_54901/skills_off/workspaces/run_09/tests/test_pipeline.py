"""Tests for the churn experiment pipeline."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_and_clean, get_features_target, get_time_splits, FEATURE_COLS, LEAK_COLS
from src.models import build_lr_pipeline, build_gb_pipeline
from src.evaluate import compute_metrics, summarise


DATA_PATH = Path(__file__).parent.parent / "churn.csv"


@pytest.fixture(scope="module")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip("churn.csv not found — run: python3 make_dataset.py --out churn.csv")
    import pandas as pd
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def clean_df():
    if not DATA_PATH.exists():
        pytest.skip("churn.csv not found — run: python3 make_dataset.py --out churn.csv")
    df, _ = load_and_clean(str(DATA_PATH))
    return df


class TestDataCleaning:
    def test_deduplication_removes_duplicates(self, raw_df):
        df, n_dropped = load_and_clean(str(DATA_PATH))
        assert n_dropped > 0, "Expected duplicates to be dropped"
        assert df.duplicated().sum() == 0, "Cleaned dataframe must have no duplicates"

    def test_leak_columns_removed(self, clean_df):
        for col in LEAK_COLS:
            assert col not in clean_df.columns, f"Leak column '{col}' must be dropped"

    def test_expected_feature_columns_present(self, clean_df):
        for col in FEATURE_COLS:
            assert col in clean_df.columns, f"Feature column '{col}' must be present"

    def test_no_missing_values(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_sorted_chronologically(self, clean_df):
        """After cleaning, rows should be sorted by signup_days ascending."""
        assert (clean_df["signup_days"].diff().dropna() >= 0).all()

    def test_churn_rate_reasonable(self, clean_df):
        rate = clean_df["churned"].mean()
        assert 0.1 <= rate <= 0.5, f"Churn rate {rate:.2%} is outside expected range"

    def test_account_status_absent(self):
        """account_status encodes target — must not appear in features."""
        df, _ = load_and_clean(str(DATA_PATH))
        assert "account_status" not in df.columns


class TestModels:
    def test_lr_pipeline_fits_and_predicts(self, clean_df):
        X, y = get_features_target(clean_df)
        pipe = build_lr_pipeline(random_state=0)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_gb_pipeline_fits_and_predicts(self, clean_df):
        X, y = get_features_target(clean_df)
        pipe = build_gb_pipeline(random_state=0)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_lr_pipeline_has_scaler(self):
        pipe = build_lr_pipeline()
        assert "scaler" in pipe.named_steps

    def test_gb_pipeline_has_no_scaler(self):
        pipe = build_gb_pipeline()
        assert "scaler" not in pipe.named_steps


class TestMetrics:
    def test_perfect_predictions_auc_one(self):
        y = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y, y_prob)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_random_predictions_auc_near_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 1000)
        y_prob = rng.random(1000)
        m = compute_metrics(y, y_prob)
        assert 0.4 <= m["roc_auc"] <= 0.6

    def test_metrics_keys_present(self):
        y = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])
        m = compute_metrics(y, y_prob)
        for key in ("roc_auc", "avg_precision", "f1", "precision", "recall"):
            assert key in m

    def test_summarise_computes_mean_std(self):
        folds = [{"roc_auc": 0.8, "f1": 0.5}, {"roc_auc": 0.9, "f1": 0.6}]
        s = summarise(folds)
        assert s["roc_auc"]["mean"] == pytest.approx(0.85)
        assert s["roc_auc"]["n"] == 2


class TestSanityChecks:
    """Verify that sanity properties hold on the real pipeline."""

    def test_label_shuffle_degrades_performance(self, clean_df):
        """Average AUC over 5 shuffle seeds must drop well below real model performance.

        A single shuffle seed can produce an unexpectedly high AUC by chance (the tiny
        residual coefficients may accidentally align with a real signal in the test set).
        Averaging across seeds gives a stable estimate that should be near chance (0.5).
        """
        from sklearn.model_selection import train_test_split

        X, y = get_features_target(clean_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        shuffled_aucs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            shuffled = y_train.copy()
            rng.shuffle(shuffled)
            pipe = build_lr_pipeline(random_state=seed)
            pipe.fit(X_train, shuffled)
            m = compute_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
            shuffled_aucs.append(m["roc_auc"])

        mean_shuffled_auc = np.mean(shuffled_aucs)
        assert mean_shuffled_auc < 0.65, (
            f"Mean shuffled-label AUC = {mean_shuffled_auc:.4f} (seeds 0-4: "
            f"{[round(a, 3) for a in shuffled_aucs]}) — still too high. Possible leak."
        )

    def test_overfit_small_batch(self, clean_df):
        """Model must achieve near-zero training loss on a tiny subset."""
        X, y = get_features_target(clean_df)
        X_tiny, y_tiny = X[:30], y[:30]

        pipe = build_gb_pipeline(random_state=0)
        pipe.fit(X_tiny, y_tiny)
        train_proba = pipe.predict_proba(X_tiny)[:, 1]
        m = compute_metrics(y_tiny, train_proba)
        assert m["roc_auc"] > 0.9, "GB should overfit a tiny batch (training AUC must be high)"

    def test_time_splits_respect_temporal_order(self, clean_df):
        """All test indices must come after train indices in each fold."""
        splits, X, y = get_time_splits(clean_df, n_splits=3)
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min(), (
                "Test fold contains indices earlier than training fold — temporal order violated"
            )
