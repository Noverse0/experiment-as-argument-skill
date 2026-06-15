"""Generate the v2 benchmark churn dataset. Deterministic for a given seed.

v2 raises the difficulty over v1 by *disguising* the leak. v1 leaked through
`account_status` ("closed" iff churned) — a column whose name screams target.
Every model from Haiku up dropped it on sight, leaving no headroom (see
benchmark/EXPERIMENTS.md exp-001..003).

Planted rigor traps in v2 (documented here, NOT in the prompt the agent sees):
- days_since_last_login: a target leak hidden behind a plausible feature name.
  A churned customer has, by definition, stopped logging in, so this value is
  recorded at/after the outcome. The signal is strong but noisy (not a perfect
  separator), so a careless pipeline gets a suspiciously high AUC instead of a
  giveaway 1.0. Catching it requires reasoning about *when* the value is known,
  not pattern-matching the column name.
- 200 exact duplicate rows appended; a random split lets them straddle.
- signup_date: temporal column; random splits ignore time.
"""
import argparse

import numpy as np
import pandas as pd


def make(seed: int = 7, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    # Honest causal signal: weak/realistic, the legitimate features.
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # Disguised leak: derived from the outcome, plausible name, noisy.
    days_since_last_login = np.where(
        churn == 1,
        rng.gamma(2.0, 22.0, n),  # mean ~44 days
        rng.gamma(2.0, 4.0, n),   # mean ~8 days
    ).round().astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "days_since_last_login": days_since_last_login,
            "churned": churn,
        }
    )
    dup = df.sample(n=200, random_state=seed)
    return pd.concat([df, dup], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="churn.csv")
    args = parser.parse_args()
    make(args.seed).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
