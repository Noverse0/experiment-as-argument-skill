import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.sanity import check_baseline_floor, check_label_shuffle, check_overfit_tiny


@pytest.fixture
def good_data():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.4 > 0).astype(int)
    return X, y


@pytest.fixture
def lr():
    return LogisticRegression(max_iter=1000, random_state=42)


def test_baseline_floor_passes_on_good_data(good_data, lr):
    X, y = good_data
    auc = check_baseline_floor(lr, X, y, threshold=0.52)
    assert auc > 0.52


def test_baseline_floor_raises_on_noise():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((400, 3))
    y = rng.integers(0, 2, 400)  # pure noise labels
    lr = LogisticRegression(max_iter=1000, random_state=42)
    with pytest.raises(AssertionError):
        check_baseline_floor(lr, X, y, threshold=0.7)


def test_label_shuffle_passes_on_good_data(good_data, lr):
    X, y = good_data
    auc = check_label_shuffle(lr, X, y, max_auc=0.65)
    assert auc <= 0.65


def test_overfit_tiny_passes_on_good_data(good_data):
    X, y = good_data
    acc = check_overfit_tiny(X, y, n=80)
    assert acc >= 0.99


def test_overfit_tiny_returns_float(good_data):
    X, y = good_data
    acc = check_overfit_tiny(X, y)
    assert isinstance(acc, float)
