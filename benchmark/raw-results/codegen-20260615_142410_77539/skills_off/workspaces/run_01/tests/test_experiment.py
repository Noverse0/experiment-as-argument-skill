"""Tests for the experiment: model training and evaluation."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

from src.pipeline import load_and_clean, prepare_features, preprocess_for_lr, preprocess_for_gb


@pytest.fixture
def sample_csv():
    """Generate a small churn dataset."""
    np.random.seed(42)
    n = 200
    data = {
        "customer_id": np.arange(1, n + 1),
        "signup_date": ["2023-01-01"] * n,
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(1, 100, n),
        "churned": np.random.binomial(1, 0.25, n),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    Path(path).unlink()


class TestLogisticRegressionTraining:
    def test_lr_trains_without_error(self, sample_csv):
        """Verify that LogisticRegression trains successfully."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_scaled, X_test_scaled = preprocess_for_lr(X_train, X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)

        assert lr is not None
        assert hasattr(lr, "coef_")

    def test_lr_produces_valid_predictions(self, sample_csv):
        """Verify that LR produces probabilities in [0, 1]."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_scaled, X_test_scaled = preprocess_for_lr(X_train, X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)

        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

        assert all(0 <= p <= 1 for p in y_pred_proba)
        assert len(y_pred_proba) == len(y_test)


class TestGradientBoostingTraining:
    def test_gb_trains_without_error(self, sample_csv):
        """Verify that GradientBoostingClassifier trains successfully."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)

        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train_gb, y_train)

        assert gb is not None
        assert hasattr(gb, "estimators_")

    def test_gb_produces_valid_predictions(self, sample_csv):
        """Verify that GB produces probabilities in [0, 1]."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)

        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train_gb, y_train)

        y_pred_proba = gb.predict_proba(X_test_gb)[:, 1]

        assert all(0 <= p <= 1 for p in y_pred_proba)
        assert len(y_pred_proba) == len(y_test)


class TestMetricsComputation:
    def test_roc_auc_computable(self, sample_csv):
        """Verify that ROC-AUC can be computed."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_scaled, X_test_scaled = preprocess_for_lr(X_train, X_test)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)

        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        assert 0 <= auc <= 1

    def test_f1_computable(self, sample_csv):
        """Verify that F1-score can be computed."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train_gb, y_train)

        y_pred = gb.predict(X_test_gb)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        assert 0 <= f1 <= 1


class TestLabelShuffleSanity:
    def test_models_fail_on_shuffled_labels(self, sample_csv):
        """Verify that models perform near-baseline on shuffled labels."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        y_shuffled = y.copy()
        np.random.RandomState(42).shuffle(y_shuffled.values)

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y_shuffled.iloc[:150], y_shuffled.iloc[150:]

        X_train_scaled, X_test_scaled = preprocess_for_lr(X_train, X_test)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)

        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # On shuffled labels, AUC should be near 0.5 (random guessing)
        # Allow some slack due to randomness, but should be < 0.65
        assert auc < 0.65, "Model should perform near-baseline on shuffled labels"


class TestOverfitSanity:
    def test_models_overfit_small_batch(self, sample_csv):
        """Verify that models can overfit a tiny batch (sanity check)."""
        df = load_and_clean(sample_csv)
        X, y, _ = prepare_features(df)

        # Take first 20 samples only
        X_tiny = X.iloc[:20]
        y_tiny = y.iloc[:20]

        X_tiny_lr, _ = preprocess_for_lr(X_tiny, X_tiny)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tiny_lr, y_tiny)

        y_pred_proba = lr.predict_proba(X_tiny_lr)[:, 1]
        auc = roc_auc_score(y_tiny, y_pred_proba)

        # Should be able to overfit to high AUC on tiny batch
        assert auc > 0.7, "Should be able to achieve high AUC on tiny batch"
