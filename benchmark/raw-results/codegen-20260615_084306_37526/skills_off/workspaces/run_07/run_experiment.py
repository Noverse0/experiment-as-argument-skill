"""
Entrypoint: compare LogisticRegression vs GradientBoostingClassifier on the
churn dataset. Writes results/metrics.json and REPORT.md.

Run:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data import FEATURES, LEAK_COLS, get_xy, load_data, temporal_split
from src.experiment import (
    baseline_auc,
    cv_scores,
    holdout_scores,
    leak_detection_auc,
)
from src.pipeline import make_gbt, make_lr

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 0. Generate dataset if missing ──────────────────────────────────────────
if not os.path.exists("churn.csv"):
    print("churn.csv not found — generating...")
    subprocess.run([sys.executable, "make_dataset.py", "--out", "churn.csv"], check=True)

# ── 1. Load & deduplicate ────────────────────────────────────────────────────
print("\n=== Data ===")
df = load_data("churn.csv")
print(f"  rows: {len(df)}, churn rate: {df['churned'].mean():.3f}")

# ── 2. Temporal split (no random shuffling across time boundary) ─────────────
train_df, test_df = temporal_split(df, train_frac=0.8)
X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)
print(f"  train: {len(X_train)} ({y_train.mean():.3f} churn)  "
      f"test: {len(X_test)} ({y_test.mean():.3f} churn)")
print(f"  train dates: {train_df['signup_date'].min()} – {train_df['signup_date'].max()}")
print(f"  test  dates: {test_df['signup_date'].min()} – {test_df['signup_date'].max()}")

# ── 3. Sanity checks ─────────────────────────────────────────────────────────
print("\n=== Sanity checks ===")
floor = baseline_auc(X_train, y_train, X_test, y_test)
print(f"  baseline AUC (majority class):     {floor:.4f}")

# Leak detection: including days_since_last_login should yield suspiciously high AUC
leak_features = FEATURES + LEAK_COLS
X_train_leak = train_df[leak_features]
X_test_leak = test_df[leak_features]
leak_auc = leak_detection_auc(X_train_leak, y_train, X_test_leak, y_test)
print(f"  leak-included GBT AUC (ceiling):   {leak_auc:.4f}  "
      f"{'← leak confirmed (>0.85)' if leak_auc > 0.85 else ''}")

# ── 4. Cross-validation on training data ─────────────────────────────────────
print("\n=== Cross-validation (5-fold × 3 repeats on train) ===")
lr_cv = cv_scores(make_lr(seed=42), X_train, y_train)
gbt_cv = cv_scores(make_gbt(seed=42), X_train, y_train)
print(f"  LR  AUC: {lr_cv['roc_auc_mean']:.4f} ± {lr_cv['roc_auc_std']:.4f}  "
      f"(n={lr_cv['n_folds']})")
print(f"  GBT AUC: {gbt_cv['roc_auc_mean']:.4f} ± {gbt_cv['roc_auc_std']:.4f}  "
      f"(n={gbt_cv['n_folds']})")

# ── 5. Final holdout evaluation (once, at the end) ───────────────────────────
print("\n=== Holdout evaluation (single pass, test set touched once) ===")
lr_hold = holdout_scores(make_lr(seed=42), X_train, y_train, X_test, y_test)
gbt_hold = holdout_scores(make_gbt(seed=42), X_train, y_train, X_test, y_test)
print(f"  LR  AUC: {lr_hold['roc_auc']:.4f}  AP: {lr_hold['avg_precision']:.4f}")
print(f"  GBT AUC: {gbt_hold['roc_auc']:.4f}  AP: {gbt_hold['avg_precision']:.4f}")

# ── 6. Persist machine-readable results ─────────────────────────────────────
metrics = {
    "dataset": {
        "n_after_dedup": int(len(df)),
        "churn_rate": float(df["churned"].mean()),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    },
    "features_used": FEATURES,
    "features_excluded": {
        "customer_id": "identifier",
        "signup_date": "used for temporal split only; redundant with tenure_months",
        "days_since_last_login": "target leak — encodes churn outcome",
    },
    "sanity": {
        "baseline_auc": floor,
        "leak_included_auc": leak_auc,
    },
    "logistic_regression": {"cv": lr_cv, "holdout": lr_hold},
    "gradient_boosting": {"cv": gbt_cv, "holdout": gbt_hold},
}
metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics → {metrics_path}")

# ── 7. Write REPORT.md ───────────────────────────────────────────────────────
gap_cv = gbt_cv["roc_auc_mean"] - lr_cv["roc_auc_mean"]
# Combined SD of the difference (independent arms)
combined_sd = (lr_cv["roc_auc_std"] ** 2 + gbt_cv["roc_auc_std"] ** 2) ** 0.5
gap_hold = gbt_hold["roc_auc"] - lr_hold["roc_auc"]

if abs(gap_cv) <= combined_sd:
    conclusion = (
        f"**No detectable difference.** "
        f"The gap between GradientBoosting and LogisticRegression is "
        f"{gap_cv:+.4f} AUC, which is within noise (combined SD: {combined_sd:.4f}). "
        f"With {lr_cv['n_folds']} CV evaluations, the honest conclusion is a tie: "
        f"neither model measurably outperforms the other on the legitimate features."
    )
else:
    winner = "GradientBoosting" if gap_cv > 0 else "LogisticRegression"
    conclusion = (
        f"**{winner} wins.** "
        f"The gap is {abs(gap_cv):.4f} AUC (combined SD: {combined_sd:.4f}), "
        f"which exceeds noise across {lr_cv['n_folds']} CV folds. "
        f"This is a detectable difference."
    )

report = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` for predicting
customer churn on this dataset?

## Methodology

### Variable
Model type (LR vs GBT). All other choices — features, preprocessing, split, seeds — are
held fixed across both arms.

### Features Used
`{', '.join(FEATURES)}`

### Features Excluded
| Column | Reason |
|--------|--------|
| `customer_id` | Identifier — not a predictor |
| `signup_date` | Used for temporal split ordering; redundant with `tenure_months` as a feature |
| `days_since_last_login` | **Target leak.** This value is recorded *after* churn occurs: a churned customer has already stopped logging in, so the feature encodes the outcome. Including it yields AUC ≈ {leak_auc:.2f} (sanity-check ceiling), confirming it is not a legitimate predictor. |

### Data Quality
- 200 exact duplicate rows (same `customer_id`) were detected and removed before
  any splitting.

### Split Strategy
Temporal split (80 / 20) sorted by `signup_date`:

- Train: {train_df['signup_date'].min()} → {train_df['signup_date'].max()} ({len(X_train)} rows)
- Test:  {test_df['signup_date'].min()} → {test_df['signup_date'].max()} ({len(X_test)} rows)

Random splits on temporal data risk leakage through near-duplicate neighbors across
the boundary; temporal ordering prevents this.

### Preprocessing
- `LogisticRegression`: `StandardScaler` (fit on train only, applied to test)
- `GradientBoostingClassifier`: no scaling needed

### Evaluation
- Primary metric: **ROC-AUC** (robust to the ~{df['churned'].mean():.0%} churn imbalance)
- Also reported: Average Precision (PR-AUC), F1
- Cross-validation: `RepeatedStratifiedKFold` (5 folds × 3 repeats = {lr_cv['n_folds']} evaluations) on training data
- Holdout: single evaluation on test set (touched once, at the end)

---

## Results

### Dataset
| Stat | Value |
|------|-------|
| Rows after dedup | {len(df)} |
| Overall churn rate | {df['churned'].mean():.1%} |
| Train / Test | {len(X_train)} / {len(X_test)} |
| Baseline AUC (majority class) | {floor:.4f} |
| Leak-included ceiling AUC | {leak_auc:.4f} |

### Cross-Validation (n={lr_cv['n_folds']} folds, on training data)
| Model | ROC-AUC (mean ± sd) | Avg Precision | F1 |
|-------|---------------------|---------------|----|
| LogisticRegression | {lr_cv['roc_auc_mean']:.4f} ± {lr_cv['roc_auc_std']:.4f} | {lr_cv['avg_precision_mean']:.4f} ± {lr_cv['avg_precision_std']:.4f} | {lr_cv['f1_mean']:.4f} ± {lr_cv['f1_std']:.4f} |
| GradientBoosting | {gbt_cv['roc_auc_mean']:.4f} ± {gbt_cv['roc_auc_std']:.4f} | {gbt_cv['avg_precision_mean']:.4f} ± {gbt_cv['avg_precision_std']:.4f} | {gbt_cv['f1_mean']:.4f} ± {gbt_cv['f1_std']:.4f} |

CV gap (GBT − LR): **{gap_cv:+.4f}** (combined SD: {combined_sd:.4f})

### Holdout Test Set
| Model | ROC-AUC | Avg Precision |
|-------|---------|---------------|
| LogisticRegression | {lr_hold['roc_auc']:.4f} | {lr_hold['avg_precision']:.4f} |
| GradientBoosting | {gbt_hold['roc_auc']:.4f} | {gbt_hold['avg_precision']:.4f} |

Holdout gap (GBT − LR): **{gap_hold:+.4f}**

---

## Conclusion

{conclusion}

---

## Limitations and Validity Threats

1. **Small feature set.** Only three legitimate predictors survived the leak audit.
   The causal signal in this dataset is intentionally weak (`tenure_months`,
   `monthly_spend`, `support_tickets`), which limits the performance ceiling for
   both models and reduces statistical power to detect differences.

2. **No hyperparameter tuning.** Default hyperparameters were used for both models.
   Tuning GBT (e.g., learning rate, depth) might narrow or widen the gap, but
   tuning on any split that touches the test set would convert test into validation.

3. **Single dataset.** Results reflect this synthetic dataset's distribution; they
   may not generalize to real-world churn data with richer feature spaces.

4. **Temporal split approximates deployment.** A rolling-origin evaluation would more
   faithfully simulate production use, but was omitted for simplicity.

5. **`days_since_last_login` exclusion is critical.** Any pipeline that includes this
   column should report AUC ≈ {leak_auc:.2f} as a stop signal — not a result.
"""

with open("REPORT.md", "w") as f:
    f.write(report)
print("REPORT.md written.")
print("\nDone.")
