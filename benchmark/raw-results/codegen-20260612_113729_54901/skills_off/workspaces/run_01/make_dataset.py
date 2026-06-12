"""Generate the benchmark churn dataset. Deterministic for a given seed.

Planted rigor traps (documented here, NOT in the prompt the agent sees):
- account_status: derived from the target ("closed" iff churned). Perfect leak.
- 200 exact duplicate rows appended; a random split lets them straddle train/test.
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
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "account_status": np.where(churn == 1, "closed", "active"),
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
