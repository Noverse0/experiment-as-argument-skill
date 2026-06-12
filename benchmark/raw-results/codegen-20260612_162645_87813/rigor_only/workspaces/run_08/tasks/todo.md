# Churn: GradientBoosting vs LogisticRegression

## Goal / Claim
Does GradientBoostingClassifier outperform LogisticRegression at predicting `churned`
on a *clean* (leak-free, deduplicated) evaluation? Report effect size with variance.

## Design decisions (the variable = model type; everything else held fixed)
- [x] Inspect dataset; audit leak surface
- Leak surface found:
  - `account_status` is "closed" iff churned==1 -> **perfect target leak -> DROP**
  - `customer_id` -> identifier, no signal -> DROP
  - `signup_date` -> temporal -> NOT a feature; used for a time-based split robustness check
  - 200 exact duplicate rows -> **dedup BEFORE splitting** so they cannot straddle train/test
- Features used: tenure_months, monthly_spend, support_tickets
- Base rate ~0.27 (imbalanced) -> primary metric ROC-AUC + average precision, not accuracy
- Preprocessing (StandardScaler) fit on train fold only, inside a Pipeline
- Repetition: RepeatedStratifiedKFold (5 folds x 3 repeats = 15 paired estimates), same folds for both models
- Robustness: single time-based split on signup_date (train early, test late)
- All seeds fixed and logged

## Sanity checks (must pass before believing results)
- [ ] majority-class baseline AUC ~ 0.5
- [ ] both models beat baseline
- [ ] leakage demo: including account_status -> AUC ~ 1.0 (proves why we dropped it)
- [ ] label-shuffle -> AUC ~ 0.5
- [ ] clean features are NOT near-perfect (AUC < 0.95)
- [ ] determinism: same seed twice -> identical metrics

## Deliverables
- [ ] src/ (data, experiment, sanity)
- [ ] run_experiment.py -> results/metrics.json + REPORT.md
- [ ] tests/ pytest
- [ ] requirements.txt
- [ ] run tests + experiment

## Results
- All 11 pytest tests pass; full experiment runs in ~5.6s (<< 5 min budget).
- Sanity: leak demo AUC=1.000, label-shuffle AUC=0.51, clean AUC<0.95, both beat baseline.
- LR ROC-AUC 0.7359±0.0133 vs GB 0.7292±0.0107 over 15 paired folds.
- GB−LR = −0.0066, 95% CI [−0.0097, −0.0036], paired p=0.0004.
- Conclusion: **No — GB does NOT outperform LR.** LR is statistically distinguishable
  but the margin is practically negligible (<0.01 AUC); direction expected since the
  data-generating process is linear.
- Artifacts: results/metrics.json + REPORT.md.
