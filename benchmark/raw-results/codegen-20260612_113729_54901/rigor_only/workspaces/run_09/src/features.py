FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "signup_days"]
TARGET_COL = "churned"


def get_Xy(df):
    """Extract feature matrix (numpy array) and target vector."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y
