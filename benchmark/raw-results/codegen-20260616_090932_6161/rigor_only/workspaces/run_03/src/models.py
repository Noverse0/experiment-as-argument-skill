"""Model definitions and training."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_logistic_regression(random_state=42):
    """Logistic regression with standard scaling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def make_gradient_boosting(random_state=42):
    """Gradient boosting classifier with minimal tuning."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fit model on train, evaluate on test.

    Returns:
        (model, metrics_dict)
    """
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    from src.metrics import evaluate

    metrics = evaluate(y_test, y_pred_proba, y_pred)
    return model, metrics
