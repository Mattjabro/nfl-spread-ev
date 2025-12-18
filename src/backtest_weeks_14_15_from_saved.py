import numpy as np
import pandas as pd
from math import erf, sqrt

TEMPERATURE = 1.75
EV_THRESHOLD = 0.05
SPREAD_PAYOUT = 0.9091


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def cover_prob_home(mu, sigma, spread_home):
    z = (mu + spread_home) / sigma
    return norm_cdf(z / TEMPERATURE)


df = pd.read_csv("../results/predictions_vs_vegas.csv")

df = df[
    (df["season"] == 2024) &
    (df["week"].isin([14, 15]))
].copy()

rows = []

for _, r in df.iterrows():
    mu = r["model_spread"]
    sigma = max(1e-6, r["sigma"])
    spread = r["closing_spread_home"]

    p_home = cover_prob_home(mu, sigma, spread)
    p_away = 1 - p_home

    ev_home = p_home * SPREAD_PAYOUT - (1 - p_home)
    ev_away = p_away * SPREAD_PAYOUT - (1 - p_away)

    if ev_home >= EV_THRESHOLD:
        bet = f"{r['home_team']} {spread}"
        ev = ev_home
        won = (r["actual_margin"] + spread) > 0
    elif ev_away >= EV_THRESHOLD:
        bet = f"{r['away_team']} {'+' if spread < 0 else '-'}{abs(spread)}"
        ev = ev_away
        won = (r["actual_margin"] + spread) < 0
    else:
        continue

    rows.append({
        "season": r["season"],
        "week": r["week"],
        "away_team": r["away_team"],
        "home_team": r["home_team"],
        "bet": bet,
        "ev": ev,
        "won": won
    })

out_df = pd.DataFrame(rows)

print("\n================ WEEKS 14–15 (2024) BACKTEST =================")
print(out_df)

print("\nSUMMARY")
print("Total bets:", len(out_df))
print("Win rate:", out_df["won"].mean())
print("Average EV:", out_df["ev"].mean())