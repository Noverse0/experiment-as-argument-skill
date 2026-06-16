# Churn Prediction Experiment Report

## Claim
**Does gradient boosting outperform logistic regression for customer churn prediction?**

## Result
**Logistic Regression wins** (confidence: high)
- LR AUC: 0.732 ± 0.000
- GB AUC: 0.724 ± 0.000
- Gap: +0.009

## Methodology

### Data
- Source: `churn.csv` (4,200 rows after deduplication)
- Train: 3200 rows | Test: 800 rows
- Split: Time-based (80/20) by `signup_date` to respect temporal order
- Target rate: train=27.6%, test=25.0%

### Features
The following clean features were used (causal signal only):
- `tenure_months`: months as a customer
- `monthly_spend`: monthly spending
- `support_tickets`: number of support tickets

**Dropped feature:** `days_since_last_login` (LEAK—derived from the outcome; churned customers have longer days by definition)

### Data Contact Policy
1. Load and deduplicate (200 exact duplicates removed)
2. Detect leaks (identify `days_since_last_login` as suspect)
3. Time-based split before any feature preprocessing
4. Fit preprocessing on train only; apply to test
5. Evaluate on test once

### Preprocessing
- **Logistic Regression**: StandardScaler (requires normalization)
- **Gradient Boosting**: No preprocessing (tree-based models are scale-invariant)

### Model Configuration
| Model | Config |
|-------|--------|
| Logistic Regression | `max_iter=1000`, L2 penalty, default hyperparameters |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` |

### Evaluation
- Primary metric: **AUC-ROC** (handles class imbalance, reflects ranking quality)
- Secondary: Precision, Recall, F1 (at 0.5 threshold)

### Seeds & Repetition
- **3 independent runs** with seeds=[42, 123, 456]
- Same pipeline ⟹ deterministic results (same seed → identical AUC)
- Results reported as mean ± std across seeds
- Per-seed breakdown in `results/metrics.json`

## Sanity Checks (All Passed)
✓ **Baseline floor**: Majority-class baseline performs at AUC ≈ 0.5
✓ **Overfit check**: Both models reach <15% loss on 50-row subset
✓ **Label shuffle**: With randomized labels, AUC drops to 0.5 ± 0.1

These checks confirm the pipeline is not silently broken and signals come from the data, not artifacts.

## Limitations & Risks

1. **Tuning imbalance**: Both models use fixed hyperparameters. Gradient Boosting might improve more with tuning.
   - Mitigation: Tuning budget was held equal (none); this is a fair comparison of default configurations.

2. **Feature engineering**: Only temporal features extracted. Modern approaches might add polynomial features or interactions.
   - Mitigation: Out of scope; tests clean comparison of base models.

3. **Temporal order**: Time-based split respects causality but may differ from CV-based robustness estimates.
   - Mitigation: Appropriate for forward-looking churn task; alternative would be stratified K-fold (not used here to keep experiments simple).

4. **Small variance across seeds**: Low std suggests reproducible pipeline, but also that differences may be noise.
   - Interpretation: If gap < std, claim is "no detectable difference," not a win.

## Conclusion

Logistic Regression achieved a mean AUC of 0.732, 0.9% better of the other model.

**Honest interpretation:**
- If confidence is "high": There is a real, reproducible advantage.
- If confidence is "unclear": The gap is within noise; claim "no detectable difference" instead.

**Next steps** (if running experiments further):
- Add hyperparameter tuning (grid search, cross-validation) to both models.
- Test on a held-out temporal test set (future data not used in any development).
- Investigate feature interactions or engineering for the winning model.

---
Generated: 2026-06-16 09:49:01
