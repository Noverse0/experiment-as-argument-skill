"""Tests for the churn prediction experiment."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data import (
    load_churn_data,
    check_duplicates,
    deduplicate,
    detect_leak_days_since_login,
    prepare_features,
    get_baseline_predictions,
    report_class_distribution,
)
from src.experiment import Experiment


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_last_login': np.random.randint(1, 100, n),
        'churned': np.random.binomial(1, 0.3, n),
    })
    return df


class TestDataLoading:
    def test_load_churn_data(self):
        """Test that churn.csv can be loaded."""
        df = load_churn_data('churn.csv')
        assert len(df) > 0
        required_cols = ['customer_id', 'signup_date', 'tenure_months', 'monthly_spend',
                        'support_tickets', 'days_since_last_login', 'churned']
        for col in required_cols:
            assert col in df.columns

    def test_load_churn_data_shape(self):
        """Test that loaded data has expected shape."""
        df = load_churn_data('churn.csv')
        assert df.shape[0] == 4200  # 4000 + 200 duplicates
        assert df.shape[1] == 7


class TestDuplicates:
    def test_check_duplicates(self):
        """Test duplicate detection."""
        df = load_churn_data('churn.csv')
        n_dups = check_duplicates(df)
        assert n_dups > 0, "Should find duplicates in the generated dataset"

    def test_deduplicate(self):
        """Test deduplication."""
        df = load_churn_data('churn.csv')
        n_before = len(df)
        df_dedup = deduplicate(df)
        n_after = len(df_dedup)
        assert n_after < n_before, "Deduplication should reduce rows"
        assert check_duplicates(df_dedup) == 0, "Should have no duplicates after deduplication"

    def test_deduplicate_preserves_features(self, sample_data):
        """Test that deduplication preserves feature values."""
        df = sample_data.copy()
        df_dedup = deduplicate(df)
        for col in ['tenure_months', 'monthly_spend', 'support_tickets']:
            assert col in df_dedup.columns


class TestLeakDetection:
    def test_leak_detection_timing_test(self, sample_data):
        """Test that timing test computes statistics correctly."""
        leak_stats = detect_leak_days_since_login(sample_data)
        assert 'churned_mean' in leak_stats
        assert 'active_mean' in leak_stats
        assert 'diff_mean' in leak_stats
        assert leak_stats['diff_mean'] == leak_stats['churned_mean'] - leak_stats['active_mean']

    def test_leak_detection_on_real_data(self):
        """Test leak detection on real churn.csv."""
        df = load_churn_data('churn.csv')
        leak_stats = detect_leak_days_since_login(df)
        # The planted leak should show much higher days_since_last_login for churned
        assert leak_stats['churned_mean'] > leak_stats['active_mean'], \
            "Churned customers should have higher days_since_last_login (leak signal)"


class TestFeaturePreparation:
    def test_prepare_features_safe(self, sample_data):
        """Test feature preparation excludes leaky feature."""
        X, y = prepare_features(sample_data, include_leaky=False)
        assert X.shape[1] == 3, "Should have 3 safe features"
        assert X.shape[0] == len(sample_data)
        assert len(y) == len(sample_data)
        assert y.dtype in [np.int64, int]

    def test_prepare_features_shapes(self, sample_data):
        """Test that X and y have compatible shapes."""
        X, y = prepare_features(sample_data)
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2
        assert y.ndim == 1

    def test_baseline_predictions(self, sample_data):
        """Test baseline prediction generation."""
        _, y = prepare_features(sample_data)
        baseline = get_baseline_predictions(y)
        assert baseline.shape == y.shape
        assert len(np.unique(baseline)) <= 2

    def test_class_distribution(self, sample_data):
        """Test class distribution reporting."""
        _, y = prepare_features(sample_data)
        dist = report_class_distribution(y)
        assert 'n_samples' in dist
        assert 'n_churned' in dist
        assert 'n_active' in dist
        assert 'churn_rate' in dist
        assert dist['n_samples'] == len(y)
        assert dist['n_churned'] + dist['n_active'] == len(y)
        assert 0 <= dist['churn_rate'] <= 1


class TestExperiment:
    def test_experiment_initialization(self, sample_data):
        """Test experiment can be initialized."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=2, n_splits=3)
        assert exp.X.shape[0] == len(y)
        assert exp.n_repeats == 2
        assert exp.n_splits == 3

    def test_sanity_check_baseline(self, sample_data):
        """Test baseline sanity check."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y)
        baseline = exp.sanity_check_baseline()
        assert 'baseline_accuracy' in baseline
        assert 'churn_rate' in baseline
        assert 0 <= baseline['baseline_accuracy'] <= 1
        assert 0 <= baseline['churn_rate'] <= 1

    def test_sanity_check_label_shuffle(self, sample_data):
        """Test label shuffle sanity check."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=1, n_splits=2)
        shuffle_check = exp.sanity_check_label_shuffle()
        assert 'lr_auc_shuffled' in shuffle_check
        assert 'gb_auc_shuffled' in shuffle_check
        # With shuffled labels, AUC should be near 0.5
        assert abs(shuffle_check['lr_auc_shuffled'] - 0.5) < 0.3, \
            "LR AUC with shuffled labels should be near 0.5"
        assert abs(shuffle_check['gb_auc_shuffled'] - 0.5) < 0.3, \
            "GB AUC with shuffled labels should be near 0.5"

    def test_sanity_check_overfit_small_batch(self, sample_data):
        """Test overfit on small batch sanity check."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y)
        overfit = exp.sanity_check_overfit_small_batch(batch_size=20)
        assert 'lr_train_accuracy' in overfit
        assert 'gb_train_accuracy' in overfit
        # On a tiny batch, model should achieve high training accuracy
        assert overfit['lr_train_accuracy'] > 0.6, "LR should overfit small batch"
        assert overfit['gb_train_accuracy'] > 0.6, "GB should overfit small batch"

    def test_run_model_comparison(self, sample_data):
        """Test that model comparison runs without error."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=2, n_splits=3)
        exp.run_model_comparison()
        assert 'model_comparison' in exp.results
        assert len(exp.results['model_comparison']) == 2
        assert 'config' in exp.results

    def test_summary_stats(self, sample_data):
        """Test summary statistics computation."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=2, n_splits=3)
        exp.run_model_comparison()
        summary = exp.compute_summary_stats()
        assert 'lr_mean_auc' in summary
        assert 'gb_mean_auc' in summary
        assert 'mean_auc_gap' in summary
        assert 0 <= summary['lr_mean_auc'] <= 1
        assert 0 <= summary['gb_mean_auc'] <= 1

    def test_seeds_logged(self, sample_data):
        """Test that seeds are properly logged."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=3, n_splits=2)
        exp.run_model_comparison()
        assert len(exp.seeds) == 3
        assert all(isinstance(s, int) for s in exp.seeds)
        assert exp.seeds == [1000, 1001, 1002]

    def test_serialization(self, sample_data):
        """Test that results can be serialized to dict."""
        X, y = prepare_features(sample_data)
        exp = Experiment(X, y, n_repeats=1, n_splits=2)
        exp.run_model_comparison()
        result_dict = exp.to_dict()
        assert 'results' in result_dict
        assert 'summary' in result_dict


class TestEndToEnd:
    def test_load_deduplicate_experiment(self):
        """End-to-end test: load, deduplicate, prepare, run mini-experiment."""
        df = load_churn_data('churn.csv')
        df = deduplicate(df)
        X, y = prepare_features(df, include_leaky=False)

        exp = Experiment(X, y, n_repeats=1, n_splits=2)
        exp.run_model_comparison()
        summary = exp.compute_summary_stats()

        assert summary['lr_mean_auc'] > 0.5, "LR should beat random baseline"
        assert summary['gb_mean_auc'] > 0.5, "GB should beat random baseline"
        assert summary['n_runs'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
