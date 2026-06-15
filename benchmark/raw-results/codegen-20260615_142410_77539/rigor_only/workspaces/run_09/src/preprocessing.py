"""Feature scaling and preprocessing."""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def create_scaler() -> StandardScaler:
    """Create a standard scaler for numerical features."""
    return StandardScaler()


def fit_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler
) -> tuple:
    """Fit scaler on train, apply to both train and test."""
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
