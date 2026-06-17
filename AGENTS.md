# Experiment as Argument

Operating rules distilled from Andrej Karpathy's "A Recipe for Training Neural Networks" and Kapoor & Narayanan's leakage taxonomy. Use these rules to avoid experiments that run but prove nothing: leaked features, single-seed winner claims, and metrics that evaporate on a clean split.

**Tradeoff:** This skill slows down quick scripts. For anything whose output will be compared, reported, or believed, pay the upfront cost so the result survives scrutiny.

## Core Idea

An experiment is an argument, not a script. The code exists to support a claim ("A beats B on task T"), and every shortcut that weakens the argument — leakage, untested assumptions, unrepeated runs — silently converts the result into noise. Code that runs and prints a number is not evidence.

## Before Training

Answer these briefly before writing experiment code:

1. **Claim:** What exact comparison or hypothesis will the final report state? One sentence.
2. **Variable:** What is the single thing being varied? Everything else must be held fixed, including tuning budget.
3. **Data contact policy:** Which rows/columns may the model see at fit time? When is the test set allowed to be touched? (Answer: once, at the end.)
4. **Leak surface:** Which features could encode the target, the future, or the split? List suspects before coding.
5. **Proof of life:** What cheap sanity check will show the pipeline works before the full run?

If you cannot answer 1–3, the experiment is not designed yet. Stop and design it.

## Data Discipline

- **Split before transform.** Any fit-like operation (scaler, vocabulary, imputation, feature selection, target encoding) happens after the split, fitted on train only, applied to the rest.
- **Hunt target leakage with the timing test, not the name.** For every feature ask: at the instant you would make the prediction, is this value already final? If it keeps moving as the outcome unfolds — a churned customer's "days since last login" grows *because* they churned — it encodes the label and must be dropped. A plausible name ("it's just login activity," "recorded pre-prediction") is not a justification; the only valid justification is "this value is provably fixed before the outcome occurs." When you cannot prove that, drop it and state the assumption. Usual offenders: status flags, closure dates, post-hoc aggregates, and recency/inactivity counters.
- **Deduplicate across the boundary.** Exact or near-duplicate rows must not straddle train/test. Check for duplicates before splitting; say in the report how many you found.
- **Respect time.** If any column is temporal and the task is forward-looking, use a time-based split. Random splits on temporal data are leakage.
- **Class balance is a fact, not a footnote.** Report the target rate; choose metrics that survive imbalance (not accuracy alone).

## Sanity Checks Before The Full Run

Run these before believing any training loop; they cost minutes and catch most silent bugs:

- **Baseline floor:** a trivial baseline (majority class, mean prediction) — your model must beat it.
- **Leakage ceiling:** if test performance looks too good (near-perfect on a noisy task), assume leakage and audit features before celebrating.
- **Overfit one batch / tiny subset:** the model must reach ~zero loss on a tiny slice. If it cannot, the pipeline is broken.
- **Label-shuffle test:** with shuffled labels, performance must fall to the baseline floor. If it does not, information is leaking around the labels.
- **Init loss check (when applicable):** loss at initialization should match the theoretical value (e.g., -log(1/C) for C balanced classes).
- **A check that cannot fail is decoration.** Every sanity check must be able to fail and must assert on its outcome — seed it, compare against the baseline floor, and stop (or loudly flag) the run on violation. A label-shuffle or overfit test that runs unseeded or asserts nothing proves nothing, however reassuring its printout looks.

## Seeds and Repetition

- Fix and log every seed (framework, numpy, data shuffling, split).
- One seed is an anecdote. Before comparing methods, run ≥3 seeds (or CV folds) per arm and report mean ± sd and n.
- A repeat only counts if it can change the result. If the split and model are deterministic, N seeds yield N identical numbers — that is fake variance, not evidence. Put the randomness where it actually moves the metric (resampled split, model init, bootstrap), or state plainly that the result is deterministic instead of dressing it up as a variance estimate.
- Identical pipelines must produce identical metrics when re-run with the same seed. If they do not, find the nondeterminism before proceeding.

## Claims Discipline

- No winner claims without variance. If the gap between arms is within noise (overlapping spreads, no test), the honest claim is "no detectable difference."
- Report effect size with its uncertainty, not just "X is better."
- The test set is touched once. Any decision made after seeing test metrics (feature change, hyperparameter, early stop) converts test into validation — say so and re-split.
- Negative and null results go in the report. Deleting failed runs is fabrication by omission.
- The report may not claim anything the code did not measure. No "robust," "significant," or "consistently" without the numbers behind it.

## Artifacts

Every run records: config (all hyperparameters), seeds, data version or generation command, code version, and resulting metrics — in a file, not in the console scrollback. Failed runs keep their artifacts.

## Experiment Ledger

One run's artifacts are not enough once you iterate. Keep a single append-only ledger (e.g. `EXPERIMENTS.md`), newest entry first, so the series stays auditable and the next step is always visible.

- **One entry per run**, fixed shape: hypothesis → setup (the single variable, seeds, n) → result → honest conclusion → next step.
- **Human writes the judgment; a script writes the numbers.** Generate the result table with code and paste it verbatim — never hand-type a metric, or the ledger drifts from the data.
- **The ledger is the progress tracker.** Each entry's "next step" seeds the next experiment; reading top to bottom shows what was tried and what is still open.
- **Keep null, failed, and corrected results.** A wrong conclusion gets a new correction entry linking the original — never a deletion. Removing a disappointing result is fabrication by omission.

Skip this only for a genuinely one-off experiment; the moment you run a second variant, start the ledger.

## Independent Review

A single reviewer — one model or one person — shares the author's blind spots. For any conclusion that will be reported or acted on, get an independent check.

- Prefer reviewers that fail differently: a second model (e.g. a different provider) or a human, not just another pass by the same model.
- **Disagreement is the signal, not the noise.** Where independent reviewers diverge is the most uncertain claim and the first place to look. Agreement is weak evidence; divergence is a flag to investigate.
- Keep authoring and reviewing in separate passes. The author's own confidence is not a review.

## Verification

- **New experiment:** run the sanity checks above, then the smallest full run that supports the claim.
- **Modified experiment:** re-run the unchanged baseline arm; if its numbers moved, the change leaked into the control.
- **Review:** check, in order — leak surface, split-before-transform, dedup, seed logging, repetition behind every comparative claim, and report/code consistency.

## Stop Signals

Stop and re-design instead of patching if:

- test metrics improve when you add a feature you cannot explain,
- performance is near-perfect on a task that should be noisy,
- the result changes materially across seeds but the report claims one winner,
- preprocessing code touches the full dataset before the split,
- the report's claim is stronger than what n runs with this variance can support.

## Response Shape

For experiment work, summarize briefly:

```text
Claim: [what the experiment argues]
Design: [variable, split policy, seeds × repeats]
Sanity: [checks run and outcomes]
Result: [mean ± sd per arm, n, and the honest conclusion]
Risk: [remaining leak surface or validity threats]
```