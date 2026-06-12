# Churn: GB vs LR experiment

## Goal / acceptance criteria
- Answer: does GradientBoosting beat LogisticRegression at predicting `churned`?
- Honest comparison with variance (n folds), not a single-seed anecdote.
- Runs CPU-only in < 5 min.

## Plan
- [x] Inspect make_dataset.py → found planted leaks
- [x] Generate + inspect churn.csv
- [ ] data.py: load, dedup, drop leaks (account_status, customer_id), order by signup_date
- [ ] models.py: LR and GB sklearn pipelines (scaler fit on train only)
- [ ] evaluate.py: TimeSeriesSplit CV, per-fold metrics, paired comparison
- [ ] sanity.py: baseline floor, leakage ceiling, label-shuffle, dedup, overfit-tiny
- [ ] run_experiment.py: orchestrate → results/metrics.json + REPORT.md
- [ ] tests/: pytest for data discipline, sanity checks, pipeline
- [ ] pyproject.toml deps
- [ ] Run tests + experiment, verify < 5 min

## Working notes / leak surface
- account_status == "closed" iff churned==1  -> PERFECT LEAK, drop
- customer_id -> identifier, drop
- 200 exact duplicate rows -> dedup BEFORE split
- signup_date temporal, churn is forward-looking -> time-based split
- churn rate ~0.27 -> imbalanced-ish, use AUC/PR-AUC not accuracy alone
- DGP is logistic in features -> LR expected competitive; likely "no detectable difference"

## Success / sanity expectations
- model AUC > 0.5 (beats baseline), but NOT ~1.0 (would mean leak)
- label-shuffle AUC ~ 0.5
- same seed -> identical metrics

## Results (verified)
- 10/10 pytest pass; full experiment runs in ~12s CPU (< 5 min budget).
- All 5 sanity checks PASS (dedup, baseline floor, leakage ceiling, label-shuffle, overfit-tiny).
- LR ROC-AUC 0.733 ± 0.025 vs GB 0.706 ± 0.019 (n=5 time folds).
- Conclusion: GB does NOT outperform LR; LR is slightly but consistently better
  (paired diff -0.027 AUC, p=0.006). Matches the logistic DGP.
- Artifacts: results/metrics.json (machine-readable) + REPORT.md.

## Lesson captured
- A single label-shuffle draw on a small fold is itself high-variance (sd ~0.13)
  and gave a false-positive "leak" at AUC 0.67. Fix: average the shuffle over
  N=20 repeats on the largest (80%) train split so the null estimate concentrates
  near 0.5. Detection signal: mean over 50 seeds was 0.50, exposing it as noise.
