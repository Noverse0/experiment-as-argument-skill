import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models import make_gradient_boosting, make_logistic_regression


@pytest.fixture
def learnable_data():
    """Dataset with a clear signal so any reasonable model gets AUC > 0.7."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 3))
    # Signal: churn if tenure low or tickets high (mirrors the real generative process)
    y = ((X[:, 0] < 0) & (X[:, 2] > 0)).astype(int)
    return X, y


@pytest.mark.parametrize("factory", [make_logistic_regression, make_gradient_boosting])
def test_predict_shape(factory, learnable_data):
    X, y = learnable_data
    model = factory(seed=42)
    model.fit(X, y)
    assert model.predict(X).shape == (len(X),)


@pytest.mark.parametrize("factory", [make_logistic_regression, make_gradient_boosting])
def test_predict_proba_shape_and_range(factory, learnable_data):
    X, y = learnable_data
    model = factory(seed=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("factory", [make_logistic_regression, make_gradient_boosting])
def test_binary_predictions(factory, learnable_data):
    X, y = learnable_data
    model = factory(seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("factory", [make_logistic_regression, make_gradient_boosting])
def test_beats_random_on_learnable_data(factory, learnable_data):
    X, y = learnable_data
    model = factory(seed=42)
    model.fit(X, y)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert auc > 0.6, f"{factory.__name__} AUC {auc:.3f} too low for learnable data"


def test_different_seeds_produce_same_lr_predictions(learnable_data):
    """LR is deterministic given same solver; different random_state shouldn't matter much."""
    X, y = learnable_data
    m1 = make_logistic_regression(seed=1)
    m2 = make_logistic_regression(seed=2)
    m1.fit(X, y)
    m2.fit(X, y)
    # LR with lbfgs converges to the global optimum regardless of seed
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), atol=1e-4
    )
