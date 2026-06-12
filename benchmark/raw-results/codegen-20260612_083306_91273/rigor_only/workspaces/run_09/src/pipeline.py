"""Data loading, preprocessing, and model training pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return features and target.

    Drops customer_id (identifier, no predictive value) and signup_date
    (temporal; using it would require careful time-based split, not simple stratified split).

    Features kept: tenure_months, monthly_spend, support_tickets, account_status.
    """
    df = pd.read_csv(path)
    # Preserve target before dropping columns
    y = df['churned'].astype(int)
    # Drop identifier and temporal columns
    X = df.drop(columns=['customer_id', 'signup_date', 'churned'])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3,
               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split before any fit operations to prevent leakage.

    Stratified split ensures test/train have similar target distributions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on train, apply to test.

    Categorical: OrdinalEncoder.
    Numeric: StandardScaler.
    This prevents leakage of statistics from test into train.
    """
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Fit encoders on train only
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    scaler = StandardScaler()

    # Encode and scale train
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols]) if categorical_cols else np.array([]).reshape(len(X_train), 0)
    X_train_numeric = scaler.fit_transform(X_train[numeric_cols]) if numeric_cols else np.array([]).reshape(len(X_train), 0)
    X_train_processed = np.hstack([X_train_encoded, X_train_numeric]) if X_train_encoded.shape[1] > 0 and X_train_numeric.shape[1] > 0 else (X_train_encoded if X_train_encoded.shape[1] > 0 else X_train_numeric)

    # Apply same transforms to test (no fitting)
    X_test_encoded = encoder.transform(X_test[categorical_cols]) if categorical_cols else np.array([]).reshape(len(X_test), 0)
    X_test_numeric = scaler.transform(X_test[numeric_cols]) if numeric_cols else np.array([]).reshape(len(X_test), 0)
    X_test_processed = np.hstack([X_test_encoded, X_test_numeric]) if X_test_encoded.shape[1] > 0 and X_test_numeric.shape[1] > 0 else (X_test_encoded if X_test_encoded.shape[1] > 0 else X_test_numeric)

    return X_train_processed, X_test_processed


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model
) -> dict:
    """Train model and evaluate on test set."""
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    }
    return metrics


def baseline_majority_class(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Majority class baseline for ROC-AUC."""
    majority_class = np.bincount(y_train).argmax()
    baseline_pred = np.full_like(y_test, majority_class, dtype=float)
    return roc_auc_score(y_test, baseline_pred)


def test_label_shuffle(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model,
    random_state: int = 42
) -> dict:
    """Sanity check: shuffle labels, model should perform at baseline.

    If performance stays high with shuffled labels, information is leaking.
    """
    np.random.seed(random_state)
    y_train_shuffled = np.random.permutation(y_train)

    model_copy = type(model)(**model.get_params())
    y_pred_proba = train_and_evaluate(X_train, X_test, y_train_shuffled, y_test, model_copy)
    return y_pred_proba
