"""Central configuration for the churn experiment.

Keeping every knob in one place makes the experiment reproducible: the report
can cite this module as the single source of truth for seeds, splits, and the
data-contact policy.
"""
from __future__ import annotations

# --- Reproducibility -------------------------------------------------------
# Single seed used for every stochastic component (model init, shuffles).
# Logged into results so a re-run is bit-for-bit comparable.
SEED = 17

# --- Schema ----------------------------------------------------------------
TARGET = "churned"

# Columns the model is ALLOWED to learn from. All numeric, no fitting needed
# to enumerate them.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]

# Identifier: carries no generalizable signal, pure row index.
ID_COLS = ["customer_id"]

# Target leak: account_status is "closed" iff churned == 1 (verified in the
# data audit). Including it yields a near-perfect score that evaporates in
# production where status is unknown at prediction time. Dropped, not used.
LEAK_COLS = ["account_status"]

# Temporal column. The task ("predict churn") is forward-looking, so we use
# this to order a chronological split instead of a random one. It is NOT used
# as a predictor (its relationship to churn must not be assumed).
TIME_COL = "signup_date"

# --- Evaluation methodology ------------------------------------------------
# Fraction of the (chronologically last) rows reserved as the held-out test
# set. Touched exactly once, at the very end.
TEST_FRACTION = 0.30

# Number of forward-chaining CV folds used on the training portion to obtain
# variance for the model comparison. >1 fold => mean +/- sd, not an anecdote.
N_CV_SPLITS = 5

# Metrics that survive class imbalance (churn rate ~27%). ROC-AUC is the
# headline; average_precision (PR-AUC) is reported alongside.
PRIMARY_METRIC = "roc_auc"
SCORING = ["roc_auc", "average_precision"]
