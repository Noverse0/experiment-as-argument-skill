import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# days_since_last_login is a target leak: churned customers have by definition
# stopped logging in, so this value is recorded after the outcome is determined.
LEAK_COLS = ["days_since_last_login"]
DROP_COLS = ["customer_id", "signup_date"] + LEAK_COLS
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]


def load_and_clean(path: str) -> tuple:
    df = pd.read_csv(path)
    # Deduplicate before any split so duplicates cannot straddle train/test.
    df = df.drop_duplicates()
    X = df[FEATURE_COLS].copy()
    y = df["churned"]
    return X, y


def build_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
    if model_name == "gradient_boosting":
        return Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
        ])
    raise ValueError(f"Unknown model: {model_name}")
