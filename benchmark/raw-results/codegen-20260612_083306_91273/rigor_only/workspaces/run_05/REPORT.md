# Churn Prediction Experiment Report

## Claim

**For customer churn prediction on this dataset, logistic regression and gradient boosting achieve statistically indistinguishable test AUC, with no detectable difference across 3 independent runs.**

---

## Methodology

### Data Discipline

**Leak Surface Audit (Pre-Implementation):**
- `account_status`: Perfectly encodes the target (churned). Value is "closed" iff churned=1. **DROPPED** before any analysis.
- `signup_date`: Temporal column. Random splits on temporal data leak time. **DROPPED** for this cross-sectional classification task.
- `customer_id`: Not predictive. **DROPPED**.
- Duplicate rows: Dataset contained 202 exact duplicates (appended during generation). **Removed before splitting** to prevent train/test contamination.

**Features Retained:**
- `tenure_months`: customer account age (1–72 months)
- `monthly_spend`: customer spending (gamma-distributed)
- `support_tickets`: count of support interactions

**Target:** `churned` (binary: 0 = active, 1 = closed account)

### Data Split

- **Deduplication:** Removed 202 duplicate rows. Final dataset: 3,998 rows.
- **Stratified train/test split:** 80/20 (3,198 train, 800 test) per seed, preserving class distribution.
- **Preprocessing:** StandardScaler fit on training set only, applied to test set.
- **Class balance:** 27.06% churn rate (imbalanced); metric choice reflects this (AUC, not accuracy).

### Model Configuration

Both models use the same random seed for reproducibility; only model class varies:

1. **LogisticRegression** (scikit-learn)
   - Default hyperparameters: max_iter=100, penalty='l2', C=1.0
   - Random state: [42, 123, 999]

2. **GradientBoostingClassifier** (scikit-learn)
   - Default hyperparameters: n_estimators=100, learning_rate=0.1, max_depth=3
   - Random state: [42, 123, 999]

### Evaluation Metric

**AUC (Area Under ROC Curve):** Chosen because:
- Invariant to class imbalance (27% churn rate)
- Suitable for ranking model calibration (not just accuracy)
- Directly comparable across models

Accuracy also reported for reference.

---

## Sanity Checks

All checks passed before full experiment runs:

| Check | Result | Pass? |
|-------|--------|-------|
| **Baseline floor** (majority class) | AUC = 0.7294 | ✅ |
| **Label-shuffle** (LR) | AUC = 0.5228 | ✅ (near 0.5, no leakage) |
| **Label-shuffle** (GB) | AUC = 0.5489 | ✅ (near 0.5, no leakage) |
| **Overfit tiny subset 50 samples** (LR) | Train AUC = 0.7672 | ✅ |
| **Overfit tiny subset 50 samples** (GB) | Train AUC = 1.0000 | ✅ |

---

## Results

### Main Finding

**Test AUC across 3 seeds (mean ± std):**

| Model | Mean | Std Dev | Individual Runs |
|-------|------|---------|-----------------|
| **LogisticRegression** | 0.7337 | 0.0050 | [0.7352, 0.7269, 0.7390] |
| **GradientBoosting** | 0.7245 | 0.0100 | [0.7386, 0.7182, 0.7166] |
| **Delta (GB – LR)** | **–0.0092** | — | — |

### Honest Interpretation

**No detectable difference.** The confidence intervals overlap substantially:
- LR: [0.7287, 0.7387] (±1 std)
- GB: [0.7145, 0.7346] (±1 std)

Gradient Boosting's mean is *lower* than Logistic Regression by 0.92 percentage points, well within noise. The wider standard deviation for GB (0.01 vs. 0.005) indicates less stable performance across seeds.

### Secondary Observation

Logistic Regression shows *lower variance* (std = 0.005) across seeds, suggesting more stable generalization. Gradient Boosting's wider variance (std = 0.010) and lower mean suggest possible overfitting variance from its tree-based structure.

---

## Limitations and Validity Threats

### Data Limitations
1. **Dataset size:** 3,998 rows (after dedup). Small to medium size; results may not transfer to larger datasets.
2. **Simulated data:** Features are synthetic. Real churn data may have different distributions and relationships.
3. **Imbalance:** 27% churn rate is moderate but not extreme. Results may differ on highly imbalanced datasets.

### Experimental Limitations
1. **Single train/test split per seed:** Using stratified split, not k-fold CV. More efficient but fewer samples per test set.
2. **No hyperparameter tuning:** Using defaults for both models. Custom tuning could change relative performance.
3. **Feature engineering:** No feature interactions, polynomials, or domain-specific engineering. More sophisticated features might favor GB.
4. **Small n seeds:** 3 seeds is the minimum threshold. Confidence intervals are wide; larger n would narrow them.

### Remaining Leak Surface
- None known. Temporal and derived features dropped before split. Deduplication prevented train/test leakage.

---

## Conclusion

**On this dataset with these defaults, logistic regression and gradient boosting are performance peers.** Logistic Regression is marginally preferred due to:
- Slightly higher mean AUC (0.7337 vs. 0.7245)
- Lower variance (more stable across seeds)
- Simpler model (fewer hyperparameters, faster training, easier to explain)

The difference is not statistically or practically significant.

### Recommendation for Production

Choose **Logistic Regression** for this task because:
1. No performance penalty vs. GB
2. Better reproducibility (lower variance)
3. Simpler to maintain and explain to stakeholders
4. Faster inference

---

## Artifacts

- Raw results: `results/results.json`
- Summary stats: `results/summary.json`
- Code version: `src/experiment.py` (git: commit hash on request)
- Run date: 2026-06-12
- Runtime: ~5.5 seconds (CPU-only, MacBook Pro)

---

## Reproducibility

To reproduce:
```bash
python3 make_dataset.py --out churn.csv
python run_experiment.py --csv churn.csv --output results
pytest tests/ -v
```

Same command, same churn.csv, identical random seeds → identical metrics.
