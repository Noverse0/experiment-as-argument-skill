# Churn Prediction Experiment Report

## Claim

**Does GradientBoostingClassifier outperform LogisticRegression for predicting customer churn?**

## Methodology

### Data
- **Source:** churn.csv (generated from make_dataset.py)
- **Total rows:** 4000 (after deduplication)
- **Train/test split:** 2800 train, 1200 test (70/30, time-based)
- **Split rationale:** Time-based split (by signup_date) avoids temporal leakage

### Preprocessing
- **Features removed:** account_status (perfect leak — derived from target)
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Scaling:** StandardScaler fitted on train, applied to test
- **No data leakage:** Verified no exact duplicate customer_ids straddle boundary

### Models
- **LogisticRegression:** max_iter=1000, default regularization
- **GradientBoostingClassifier:** n_estimators=100, max_depth=3, learning_rate=0.1

### Sanity Checks Performed
1. **Baseline floor:** Majority class classifier achieves 0.5000 AUC
   - Both models must beat this
2. **Tiny overfit check:** Model must fit ~zero loss on 50-row subset (n=50)
   - Passed: GradientBoosting AUC > 0.95 on tiny set
3. **Label shuffle test:** With shuffled labels, performance must be random
   - Verified information does not leak around labels

### Experiment Design
- **Variable:** Model type (single variable changed)
- **Repetitions:** 5 seeds (42, 123, 456, 789, 999) for variance
- **Metrics:** AUC, precision, recall, F1 at threshold=0.5
- **All other factors held fixed:** train/test split, features, hyperparameters

## Results

```
LogisticRegression AUC: 0.7459 ± 0.0000
GradientBoosting AUC: 0.7347 ± 0.0000
Conclusion: logistic regression outperforms by 0.0112 AUC
```

**Detailed metrics:** See results/metrics.json (mean ± std per metric, all values per seed)

## Interpretation

- **Overlapping confidence intervals:** If ± bands overlap, the difference is within noise
- **Effect size:** Report the actual AUC difference with ±std
- **Multiple comparisons:** Only one claim being tested (GB vs LR), no multiple testing

## Limitations & Risk

1. **Hyperparameter tuning:** Models use fixed hyperparameters (not tuned on validation set)
   - A tuned GB might perform better or worse; claim is about defaults
2. **Preprocessing scope:** Only StandardScaler + feature selection (drop leaky column)
   - More sophisticated feature engineering might change the conclusion
3. **Small dataset:** 4000 original rows; variance may be high
   - Recommendation: repeat on larger datasets if generalizing this result
4. **Single churn prediction task:** Cannot claim generalization beyond this domain

## Reproducibility

```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
pytest tests/
```

- Experiment is deterministic given seed
- All seeds, split cutoffs, and config logged in code
- Results saved to results/metrics.json and REPORT.md
