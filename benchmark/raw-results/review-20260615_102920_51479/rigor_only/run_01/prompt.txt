# experiment-as-argument-review-v2

You are reviewing a generated ML experiment project in the current working directory. Judge how rigorous the experiment is as evidence, not how pretty the code is.

This is the v2 dataset, whose leak is disguised. Read the leakage guidance carefully before scoring.

## Procedure

1. Inventory the files. Read the experiment code, tests, and REPORT.md.
2. Run the declared test command and the declared entrypoint. If a command would clearly exceed 120 seconds, do not wait it out — record that fact and score from static evidence instead.
3. Audit, in order:
   - **Leakage (the central test):** Is `days_since_last_login` used as a model feature? It is a **target leak in disguise**: a churned customer has by definition stopped logging in, so this value is recorded at or after the outcome, not before the prediction point. Using it is a leak even though its name looks like a legitimate behavioral feature. Signs the project caught it: the feature is dropped or explicitly justified as pre-outcome; or the report flags that including it produces a suspiciously high AUC (~0.95 vs ~0.73 on the honest features) and treats that as a leakage signal, not a win. A project that simply uses it and reports ~0.95 without question has leaked and should score low on leakage prevention regardless of how clean the code is. Also check the usual fit-on-train-only discipline (scaler/imputer/encoders fitted on train split only), whether duplicate rows straddle the train/test split, and whether the temporal column `signup_date` is handled deliberately (time-based split or explicit justification).
   - **Methodology:** Trivial baseline present? Metrics appropriate for the class balance (not accuracy alone)? Same tuning budget for both models?
   - **Reproducibility:** Seeds fixed and logged? Config/metrics written to results/? Re-running with the same seed reproducible?
   - **Claims:** Does REPORT.md claim a winner? Backed by how many runs/folds, with what variance? Are limitations honest? Does the report claim anything the code did not measure? Does it notice if its own headline number is implausibly high?
   - **Executability:** Do the declared commands actually work from a clean state?
   - **Code quality:** Readable structure, no dead code, dependencies declared and used.

## Output contract

End your review with exactly one fenced JSON block (```json ... ```) and nothing after it. The JSON must contain ALL of these fields:

```json
{
  "rubric_version": "experiment-as-argument-review-v2",
  "score_scale": "0-100 per category",
  "weights": {
    "leakage_prevention": 0.25,
    "methodological_validity": 0.20,
    "reproducibility": 0.15,
    "claims_discipline": 0.15,
    "executability": 0.15,
    "code_quality": 0.10
  },
  "leakage_prevention_score": 0,
  "methodological_validity_score": 0,
  "reproducibility_score": 0,
  "claims_discipline_score": 0,
  "executability_score": 0,
  "code_quality_score": 0,
  "verdict": "excellent|good|mixed|poor",
  "key_findings": ["..."],
  "leakage_findings": ["..."],
  "commands_run": ["..."],
  "file_evidence": [{"path": "...", "note": "..."}]
}
```

Scores are integers 0-100. Do not output a weighted total; it is recomputed downstream. Do not wrap the JSON in any other code fence or prose after it.
