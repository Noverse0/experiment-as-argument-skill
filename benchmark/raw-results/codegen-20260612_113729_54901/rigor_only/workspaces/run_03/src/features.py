from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "days_since_start"]
TARGET_COL = "churned"


def make_scaler() -> StandardScaler:
    return StandardScaler()
