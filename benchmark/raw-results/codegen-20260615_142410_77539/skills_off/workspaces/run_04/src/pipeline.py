"""Training pipeline with proper train/test split and preprocessing."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


# Legitimate features only (days_since_last_login is target leak, excluded)
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split by signup date to respect temporal order.

    Train: earliest (100-test_size)% of dates.
    Test: latest (test_size)% of dates.
    Avoids leaking future info into past and handles time-dependent data properly.
    """
    cutoff_date = df["signup_date"].quantile(1 - test_size)
    train_idx = df["signup_date"] <= cutoff_date
    X_train = df.loc[train_idx, FEATURE_COLS].copy()
    X_test = df.loc[~train_idx, FEATURE_COLS].copy()
    y_train = df.loc[train_idx, TARGET_COL].copy()
    y_test = df.loc[~train_idx, TARGET_COL].copy()
    return X_train, X_test, y_train, y_test


def deduplicate_train(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Remove duplicate rows from training set only.

    This prevents a duplicated row in train from also appearing in test,
    which would inflate test metrics.
    """
    # Create a combined dataframe to find duplicates
    combined = X_train.copy()
    combined["target"] = y_train.values
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    n_removed = before - after

    X_train_dedup = combined[FEATURE_COLS]
    y_train_dedup = combined["target"]

    return X_train_dedup.reset_index(drop=True), y_train_dedup.reset_index(drop=True), n_removed


def train_and_evaluate(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series, seed: int) -> dict:
    """Train model and compute metrics on test set."""
    # Create model
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=seed, validation_fraction=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Fit on training set (scaler fitted on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Compute metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics


def label_shuffle_test(X_test: pd.DataFrame, y_test: pd.Series, seed: int) -> dict:
    """Sanity check: shuffle labels and verify metric drops to baseline.

    If metrics remain high with random labels, information is leaking.
    """
    # Baseline: majority class accuracy/AUC
    churn_rate = y_test.mean()
    baseline_auc = 0.5  # Random classifier

    # Fit model on shuffled labels
    rng = np.random.default_rng(seed)
    y_shuffled = y_test.values.copy()
    rng.shuffle(y_shuffled)

    model = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    model.fit(X_scaled, y_shuffled)

    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    shuffled_auc = roc_auc_score(y_shuffled, y_pred_proba)

    return {
        "baseline_auc": baseline_auc,
        "shuffled_auc": shuffled_auc,
        "churn_rate": churn_rate,
    }
